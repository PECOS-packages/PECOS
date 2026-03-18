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

//! Conditions for branching in noise decision trees.

use crate::command::GateType;
use crate::noise::NoiseContext;
use pecos_core::QubitId;

/// A condition that can be evaluated against noise context.
///
/// Conditions are used in `When` primitives to branch the decision tree
/// based on qubit state.
pub trait Condition: Send + Sync {
    /// Evaluate the condition for a specific qubit.
    fn evaluate(&self, qubit: QubitId, ctx: &NoiseContext) -> bool;

    /// Human-readable name for visualization.
    fn name(&self) -> &'static str;
}

/// Condition: qubit is in leaked state.
#[derive(Debug, Clone, Copy, Default)]
pub struct Leaked;

impl Condition for Leaked {
    fn evaluate(&self, qubit: QubitId, ctx: &NoiseContext) -> bool {
        ctx.is_leaked(qubit)
    }

    fn name(&self) -> &'static str {
        "leaked"
    }
}

/// Condition: qubit is not in leaked state.
#[derive(Debug, Clone, Copy, Default)]
pub struct NotLeaked;

impl Condition for NotLeaked {
    fn evaluate(&self, qubit: QubitId, ctx: &NoiseContext) -> bool {
        !ctx.is_leaked(qubit)
    }

    fn name(&self) -> &'static str {
        "not_leaked"
    }
}

/// Condition: qubit is active (has been prepared and not yet measured).
#[derive(Debug, Clone, Copy, Default)]
pub struct Active;

impl Condition for Active {
    fn evaluate(&self, qubit: QubitId, ctx: &NoiseContext) -> bool {
        ctx.is_active(qubit)
    }

    fn name(&self) -> &'static str {
        "active"
    }
}

/// Condition: always true.
#[derive(Debug, Clone, Copy, Default)]
pub struct Always;

impl Condition for Always {
    fn evaluate(&self, _qubit: QubitId, _ctx: &NoiseContext) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "always"
    }
}

/// Condition: always false.
#[derive(Debug, Clone, Copy, Default)]
pub struct Never;

impl Condition for Never {
    fn evaluate(&self, _qubit: QubitId, _ctx: &NoiseContext) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "never"
    }
}

/// Condition: current measurement outcome equals a specific value.
///
/// This is used during measurement noise processing to apply
/// different noise based on the outcome (e.g., asymmetric readout error).
#[derive(Debug, Clone, Copy)]
pub struct OutcomeIs {
    expected: bool,
}

impl OutcomeIs {
    /// Create a condition that checks if the outcome equals the expected value.
    #[must_use]
    pub fn new(expected: bool) -> Self {
        Self { expected }
    }

    /// Check if outcome is 0.
    #[must_use]
    pub fn zero() -> Self {
        Self::new(false)
    }

    /// Check if outcome is 1.
    #[must_use]
    pub fn one() -> Self {
        Self::new(true)
    }
}

impl Condition for OutcomeIs {
    fn evaluate(&self, _qubit: QubitId, ctx: &NoiseContext) -> bool {
        ctx.current_outcome() == Some(self.expected)
    }

    fn name(&self) -> &'static str {
        if self.expected {
            "outcome_is_1"
        } else {
            "outcome_is_0"
        }
    }
}

/// Condition: current gate type matches a specific type.
///
/// This is used to apply different noise based on the gate type
/// (e.g., special handling for `MeasureLeaked`).
#[derive(Debug, Clone, Copy)]
pub struct GateTypeIs {
    expected: GateType,
}

impl GateTypeIs {
    /// Create a condition that checks if the gate type equals the expected value.
    #[must_use]
    pub fn new(expected: GateType) -> Self {
        Self { expected }
    }
}

impl Condition for GateTypeIs {
    fn evaluate(&self, _qubit: QubitId, ctx: &NoiseContext) -> bool {
        ctx.current_gate()
            .is_some_and(|info| info.gate_type == self.expected)
    }

    fn name(&self) -> &'static str {
        "gate_type_is"
    }
}

/// Condition: the OTHER qubit in a two-qubit gate is leaked.
///
/// This is used to apply partner depolarizing: when one qubit of a 2Q gate
/// is leaked, the non-leaked partner should receive depolarizing noise.
///
/// Returns `false` if:
/// - This is not a two-qubit gate
/// - The other qubit is not leaked
/// - This qubit itself is leaked (both are leaked)
///
/// # Example
///
/// ```
/// use pecos_neo::noise::composite::prelude::*;
///
/// // If partner is leaked (and I'm not), apply random Pauli to me
/// let partner_effect = when(partner_leaked(), pauli(), nothing());
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct PartnerLeaked;

impl Condition for PartnerLeaked {
    fn evaluate(&self, qubit: QubitId, ctx: &NoiseContext) -> bool {
        // Get the other qubit in this two-qubit gate
        let Some(other) = ctx.other_qubit() else {
            return false;
        };

        // I must not be leaked (only non-leaked partners get depolarized)
        if ctx.is_leaked(qubit) {
            return false;
        }

        // Partner must be leaked
        ctx.is_leaked(other)
    }

    fn name(&self) -> &'static str {
        "partner_leaked"
    }
}

/// Condition: any qubit in the current gate (including this one) is leaked.
///
/// This is a fast check that returns true if ANY qubit involved in the
/// current gate is leaked. Useful for deciding whether to skip the entire
/// gate application.
#[derive(Debug, Clone, Copy, Default)]
pub struct AnyQubitLeaked;

impl Condition for AnyQubitLeaked {
    fn evaluate(&self, _qubit: QubitId, ctx: &NoiseContext) -> bool {
        // Fast path: if no leakage at all, return false immediately
        if ctx.leaked_count() == 0 {
            return false;
        }

        // Check all qubits in the current gate
        ctx.any_leaked(ctx.current_gate_qubits())
    }

    fn name(&self) -> &'static str {
        "any_qubit_leaked"
    }
}

// ============================================================================
// Two-Stage "Fired" Conditions
// ============================================================================

/// Condition: the current qubit "fired" in stage 1.
///
/// Used in two-stage processing for correlated effects. Stage 1 samples
/// whether each qubit fires (e.g., spontaneous emission), and stage 2
/// uses this condition to check if the current qubit fired.
#[derive(Debug, Clone, Copy, Default)]
pub struct IFired;

impl Condition for IFired {
    fn evaluate(&self, _qubit: QubitId, ctx: &NoiseContext) -> bool {
        ctx.current_qubit_fired()
    }

    fn name(&self) -> &'static str {
        "i_fired"
    }
}

/// Condition: the partner qubit "fired" in stage 1 (two-qubit gates only).
///
/// Returns true if the other qubit in a two-qubit gate fired during stage 1.
/// Returns false if this is not a two-qubit gate.
#[derive(Debug, Clone, Copy, Default)]
pub struct PartnerFired;

impl Condition for PartnerFired {
    fn evaluate(&self, _qubit: QubitId, ctx: &NoiseContext) -> bool {
        ctx.partner_fired()
    }

    fn name(&self) -> &'static str {
        "partner_fired"
    }
}

/// Condition: partner fired exclusively (partner fired, I did not).
///
/// This is the key condition for partner depolarizing in two-qubit gates:
/// when one qubit emits/leaks but the other doesn't, the non-emitting
/// qubit should receive depolarizing noise.
#[derive(Debug, Clone, Copy, Default)]
pub struct PartnerOnlyFired;

impl Condition for PartnerOnlyFired {
    fn evaluate(&self, _qubit: QubitId, ctx: &NoiseContext) -> bool {
        ctx.partner_fired() && !ctx.current_qubit_fired()
    }

    fn name(&self) -> &'static str {
        "partner_only_fired"
    }
}

/// Condition from a function.
pub struct FnCondition<F>
where
    F: Fn(QubitId, &NoiseContext) -> bool + Send + Sync,
{
    func: F,
    name: &'static str,
}

impl<F> FnCondition<F>
where
    F: Fn(QubitId, &NoiseContext) -> bool + Send + Sync,
{
    /// Create a new function-based condition.
    pub fn new(name: &'static str, func: F) -> Self {
        Self { func, name }
    }
}

impl<F> Condition for FnCondition<F>
where
    F: Fn(QubitId, &NoiseContext) -> bool + Send + Sync,
{
    fn evaluate(&self, qubit: QubitId, ctx: &NoiseContext) -> bool {
        (self.func)(qubit, ctx)
    }

    fn name(&self) -> &'static str {
        self.name
    }
}

/// Convenience functions for creating conditions.
pub mod conditions {
    use super::{
        Active, AnyQubitLeaked, FnCondition, GateType, GateTypeIs, IFired, Leaked, NoiseContext,
        NotLeaked, OutcomeIs, PartnerFired, PartnerLeaked, PartnerOnlyFired, QubitId,
    };

    /// Condition: qubit is leaked.
    #[must_use]
    pub fn leaked() -> Leaked {
        Leaked
    }

    /// Condition: qubit is not leaked.
    #[must_use]
    pub fn not_leaked() -> NotLeaked {
        NotLeaked
    }

    /// Condition: qubit is active (prepared but not measured).
    #[must_use]
    pub fn active() -> Active {
        Active
    }

    /// Condition: current measurement outcome equals the given value.
    #[must_use]
    pub fn outcome_is(value: bool) -> OutcomeIs {
        OutcomeIs::new(value)
    }

    /// Condition: current measurement outcome is 0.
    #[must_use]
    pub fn outcome_is_zero() -> OutcomeIs {
        OutcomeIs::zero()
    }

    /// Condition: current measurement outcome is 1.
    #[must_use]
    pub fn outcome_is_one() -> OutcomeIs {
        OutcomeIs::one()
    }

    /// Condition: current gate type matches the expected type.
    #[must_use]
    pub fn gate_type_is(gate_type: GateType) -> GateTypeIs {
        GateTypeIs::new(gate_type)
    }

    /// Custom condition from a function.
    pub fn custom<F>(name: &'static str, f: F) -> FnCondition<F>
    where
        F: Fn(QubitId, &NoiseContext) -> bool + Send + Sync,
    {
        FnCondition::new(name, f)
    }

    // ========================================================================
    // Two-qubit gate conditions
    // ========================================================================

    /// Condition: the OTHER qubit in a two-qubit gate is leaked.
    ///
    /// Returns true if:
    /// - This is a two-qubit gate
    /// - The other qubit is leaked
    /// - This qubit is NOT leaked
    ///
    /// Use this for partner depolarizing: apply noise to non-leaked qubit
    /// when its partner is leaked.
    ///
    /// # Example
    ///
    /// ```
    /// # use pecos_neo::noise::composite::prelude::*;
    /// // If partner is leaked, apply Pauli to me
    /// when(partner_leaked(), pauli(), nothing());
    /// ```
    #[must_use]
    pub fn partner_leaked() -> PartnerLeaked {
        PartnerLeaked
    }

    /// Condition: any qubit in the current gate is leaked.
    ///
    /// Returns true if ANY qubit (including this one) in the current
    /// gate is leaked. This is a fast check useful for deciding whether
    /// to skip gate processing entirely.
    #[must_use]
    pub fn any_qubit_leaked() -> AnyQubitLeaked {
        AnyQubitLeaked
    }

    // ========================================================================
    // Two-stage "fired" conditions
    // ========================================================================

    /// Condition: the current qubit fired in stage 1.
    ///
    /// Used for two-stage composite processing. Stage 1 samples and records
    /// whether each qubit "fired" (e.g., emission event occurred).
    /// Stage 2 uses this to apply effects based on what fired.
    #[must_use]
    pub fn i_fired() -> IFired {
        IFired
    }

    /// Condition: the partner qubit fired in stage 1.
    ///
    /// Returns true if the other qubit in a two-qubit gate fired.
    /// Returns false for single-qubit gates.
    #[must_use]
    pub fn partner_fired() -> PartnerFired {
        PartnerFired
    }

    /// Condition: partner fired exclusively (partner fired, I did not).
    ///
    /// This is the key condition for partner depolarizing in two-qubit gates.
    /// When one qubit emits but the other doesn't, the non-emitting
    /// qubit should receive depolarizing noise.
    ///
    /// # Example
    ///
    /// ```
    /// # use pecos_neo::noise::composite::prelude::*;
    /// // Stage 2: if partner emitted but I didn't, depolarize me
    /// when(partner_only_fired(), pauli(), nothing());
    /// ```
    #[must_use]
    pub fn partner_only_fired() -> PartnerOnlyFired {
        PartnerOnlyFired
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_context() -> NoiseContext {
        NoiseContext::new()
    }

    #[test]
    fn test_leaked_condition() {
        let mut ctx = make_test_context();
        let cond = Leaked;

        assert!(!cond.evaluate(QubitId(0), &ctx));

        ctx.mark_leaked(QubitId(0));
        assert!(cond.evaluate(QubitId(0), &ctx));
        assert!(!cond.evaluate(QubitId(1), &ctx));
    }

    #[test]
    fn test_not_leaked_condition() {
        let mut ctx = make_test_context();
        let cond = NotLeaked;

        assert!(cond.evaluate(QubitId(0), &ctx));

        ctx.mark_leaked(QubitId(0));
        assert!(!cond.evaluate(QubitId(0), &ctx));
    }

    #[test]
    fn test_always_never() {
        let ctx = make_test_context();

        assert!(Always.evaluate(QubitId(0), &ctx));
        assert!(!Never.evaluate(QubitId(0), &ctx));
    }

    #[test]
    fn test_fn_condition() {
        let ctx = make_test_context();
        let cond = FnCondition::new("qubit_0", |q, _| q.0 == 0);

        assert!(cond.evaluate(QubitId(0), &ctx));
        assert!(!cond.evaluate(QubitId(1), &ctx));
    }

    // ========================================================================
    // Two-Stage "Fired" Condition Tests
    // ========================================================================

    #[test]
    fn test_i_fired_condition() {
        let mut ctx = make_test_context();
        let qubits = [QubitId(0), QubitId(1)];

        // Initially nothing fired
        ctx.set_current_qubit_index(0, &qubits);
        assert!(!IFired.evaluate(QubitId(0), &ctx));

        // Set qubit 0 as fired
        ctx.set_fired(0, true);
        assert!(IFired.evaluate(QubitId(0), &ctx));

        // Check qubit 1 (not fired)
        ctx.set_current_qubit_index(1, &qubits);
        assert!(!IFired.evaluate(QubitId(1), &ctx));

        // Set qubit 1 as fired too
        ctx.set_fired(1, true);
        assert!(IFired.evaluate(QubitId(1), &ctx));
    }

    #[test]
    fn test_partner_fired_condition() {
        let mut ctx = make_test_context();
        let qubits = [QubitId(0), QubitId(1)];

        // Set up as qubit 0
        ctx.set_current_qubit_index(0, &qubits);

        // Partner (qubit 1) has not fired
        assert!(!PartnerFired.evaluate(QubitId(0), &ctx));

        // Set partner as fired
        ctx.set_fired(1, true);
        assert!(PartnerFired.evaluate(QubitId(0), &ctx));

        // Switch to qubit 1 perspective
        ctx.set_current_qubit_index(1, &qubits);

        // Partner (qubit 0) has not fired
        assert!(!PartnerFired.evaluate(QubitId(1), &ctx));

        // Set qubit 0 as fired
        ctx.set_fired(0, true);
        assert!(PartnerFired.evaluate(QubitId(1), &ctx));
    }

    #[test]
    fn test_partner_only_fired_condition() {
        let mut ctx = make_test_context();
        let qubits = [QubitId(0), QubitId(1)];

        // === Scenario 1: Neither fired ===
        ctx.set_current_qubit_index(0, &qubits);
        assert!(!PartnerOnlyFired.evaluate(QubitId(0), &ctx));

        // === Scenario 2: Only partner fired ===
        ctx.set_fired(1, true);
        ctx.set_fired(0, false);
        ctx.set_current_qubit_index(0, &qubits);
        // From qubit 0's perspective: partner (qubit 1) fired, I didn't
        assert!(PartnerOnlyFired.evaluate(QubitId(0), &ctx));

        // From qubit 1's perspective: I fired, partner didn't
        ctx.set_current_qubit_index(1, &qubits);
        assert!(!PartnerOnlyFired.evaluate(QubitId(1), &ctx));

        // === Scenario 3: Both fired ===
        ctx.set_fired(0, true);
        ctx.set_fired(1, true);
        ctx.set_current_qubit_index(0, &qubits);
        assert!(!PartnerOnlyFired.evaluate(QubitId(0), &ctx));
        ctx.set_current_qubit_index(1, &qubits);
        assert!(!PartnerOnlyFired.evaluate(QubitId(1), &ctx));

        // === Scenario 4: Only I fired ===
        ctx.set_fired(0, true);
        ctx.set_fired(1, false);
        ctx.set_current_qubit_index(0, &qubits);
        assert!(!PartnerOnlyFired.evaluate(QubitId(0), &ctx));
        ctx.set_current_qubit_index(1, &qubits);
        assert!(PartnerOnlyFired.evaluate(QubitId(1), &ctx));
    }

    #[test]
    fn test_partner_fired_single_qubit_gate() {
        let mut ctx = make_test_context();

        // Single-qubit gate (only one qubit)
        ctx.set_current_qubit_index(0, &[QubitId(0)]);
        ctx.set_fired(0, true);

        // Partner conditions should return false for single-qubit gates
        assert!(!PartnerFired.evaluate(QubitId(0), &ctx));
        assert!(!PartnerOnlyFired.evaluate(QubitId(0), &ctx));
    }
}
