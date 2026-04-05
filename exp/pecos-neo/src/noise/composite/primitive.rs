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

//! Noise primitives for composable decision trees.
//!
//! Primitives are the building blocks of noise decision trees. They compose
//! together to form complex noise models while maintaining a simple, fixed API.

use std::fmt::Write as _;

use super::action::GateAction;
use super::condition::Condition;
use super::response::CompositeResponse;
use crate::noise::NoiseContext;
use pecos_core::QubitId;
use pecos_random::PecosRng;
use rand::RngExt;

/// A noise primitive that can be composed into decision trees.
///
/// Primitives form the nodes of noise decision trees. They can either be
/// control flow (branching, probability gates) or terminal actions.
pub trait Primitive: Send + Sync {
    /// Apply this primitive for a specific qubit.
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse;

    /// Human-readable description for visualization (single line).
    fn describe(&self) -> String;

    /// Clone this primitive into a boxed trait object.
    fn clone_box(&self) -> Box<dyn Primitive>;

    /// Multi-line tree representation for debugging.
    ///
    /// Returns a tree-formatted string showing the structure of composed primitives.
    /// Default implementation returns the single-line `describe()`.
    fn describe_tree(&self) -> String {
        self.describe()
    }

    /// Tree representation with indentation prefix.
    ///
    /// Used internally for nested tree rendering.
    fn describe_tree_with_prefix(&self, prefix: &str, is_last: bool) -> String {
        let connector = if is_last { "└─ " } else { "├─ " };
        format!("{}{}{}", prefix, connector, self.describe())
    }

    /// Returns true if this primitive requires two-pass processing.
    ///
    /// Two-pass primitives are processed as:
    /// 1. Stage 1: `apply_stage1` called on ALL qubits (sampling)
    /// 2. Stage 2: `apply_stage2` called on ALL qubits (effects)
    ///
    /// This enables cross-qubit conditions like "if partner fired".
    fn needs_two_pass(&self) -> bool {
        false
    }

    /// Stage 1 of two-pass processing: sampling phase.
    ///
    /// Override this to sample events and store results in fired flags.
    /// Default implementation does nothing.
    fn apply_stage1(
        &self,
        _qubit: QubitId,
        _ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        CompositeResponse::None
    }

    /// Stage 2 of two-pass processing: effect phase.
    ///
    /// Override this to apply effects based on cross-qubit conditions.
    /// Default implementation calls `apply`.
    fn apply_stage2(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        self.apply(qubit, ctx, rng)
    }
}

// ============================================================================
// Two-Stage Primitive (for correlated multi-qubit effects)
// ============================================================================

/// Result of sampling stage - stored for each qubit.
///
/// Part of the `TwoStagePrimitive` infrastructure for advanced two-stage processing.
/// Currently, the simpler `fired_flags` approach in `NoiseContext` is used instead.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, Default)]
pub struct SampleResult {
    /// Whether an event occurred for this qubit.
    pub occurred: bool,
    /// Optional Pauli type if a Pauli error was sampled.
    pub pauli: Option<crate::command::GateType>,
}

/// Storage for multi-qubit sample results.
///
/// Used by two-stage primitives to store sampling decisions before
/// applying cross-effects.
///
/// Part of the `TwoStagePrimitive` infrastructure for advanced two-stage processing.
/// Currently, the simpler `fired_flags` approach in `NoiseContext` is used instead.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct MultiQubitSamples {
    /// Sample results indexed by qubit position in gate (0, 1, ...).
    pub results: [SampleResult; 4], // Support up to 4-qubit gates
}

#[allow(dead_code)]
impl MultiQubitSamples {
    /// Get result for qubit at given index.
    #[must_use]
    pub fn get(&self, index: usize) -> SampleResult {
        self.results.get(index).copied().unwrap_or_default()
    }

    /// Set result for qubit at given index.
    pub fn set(&mut self, index: usize, result: SampleResult) {
        if index < self.results.len() {
            self.results[index] = result;
        }
    }

    /// Check if any qubit had an event occur.
    #[must_use]
    pub fn any_occurred(&self, num_qubits: usize) -> bool {
        self.results.iter().take(num_qubits).any(|r| r.occurred)
    }
}

/// A two-stage primitive for handling correlated multi-qubit effects.
///
/// Two-stage primitives process qubits in two passes:
/// 1. **Sample stage**: For each qubit, sample whether an event occurs
/// 2. **Effect stage**: For each qubit, apply effects based on ALL sample results
///
/// This enables proper handling of cross-conditions like:
/// "if qubit A leaked but qubit B didn't, depolarize B"
///
/// # Example Implementation
///
/// ```text
/// use pecos_neo::noise::composite::prelude::*;
/// struct EmissionWithPartnerDepolarize { prob: f64 }
///
/// impl TwoStagePrimitive for EmissionWithPartnerDepolarize {
///     fn sample(&self, qubit: QubitId, index: usize, ctx: &NoiseContext, rng: &mut PecosRng) -> SampleResult {
///         SampleResult {
///             occurred: rng.random::<f64>() < self.prob,
///             pauli: None,
///         }
///     }
///
///     fn apply_effects(&self, qubit: QubitId, index: usize, samples: &MultiQubitSamples,
///                      num_qubits: usize, ctx: &mut NoiseContext, rng: &mut PecosRng) -> CompositeResponse {
///         let my_result = samples.get(index);
///         let other_index = 1 - index; // For 2-qubit gates
///         let other_result = samples.get(other_index);
///
///         if my_result.occurred {
///             ctx.mark_leaked(qubit);
///             CompositeResponse::Leak
///         } else if other_result.occurred {
///             // Partner emitted, I didn't → depolarize me
///             CompositeResponse::InjectGates(vec![random_pauli_gate(qubit, rng)])
///         } else {
///             CompositeResponse::None
///         }
///     }
/// }
/// ```
#[allow(dead_code)]
pub trait TwoStagePrimitive: Send + Sync {
    /// Stage 1: Sample whether an event occurs for this qubit.
    ///
    /// Called once per qubit before any effects are applied.
    /// Results are stored and made available in `apply_effects`.
    fn sample(
        &self,
        qubit: QubitId,
        index: usize,
        ctx: &NoiseContext,
        rng: &mut PecosRng,
    ) -> SampleResult;

    /// Stage 2: Apply effects based on all sample results.
    ///
    /// Called once per qubit after all qubits have been sampled.
    /// Has access to sample results for ALL qubits in the gate.
    fn apply_effects(
        &self,
        qubit: QubitId,
        index: usize,
        samples: &MultiQubitSamples,
        num_qubits: usize,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse;

    /// Human-readable description.
    fn describe(&self) -> String;
}

// ============================================================================
// TwoStage Primitive Wrapper
// ============================================================================

/// A two-stage primitive that wraps two sub-primitives for two-pass processing.
///
/// This primitive enables proper handling of cross-qubit conditions in
/// multi-qubit gates. It processes qubits in two passes:
///
/// 1. **Stage 1**: Run `stage1` on ALL qubits (typically sampling/firing)
/// 2. **Stage 2**: Run `stage2` on ALL qubits (typically applying cross-effects)
///
/// The key use case is emission with partner depolarizing:
/// - Stage 1 samples whether each qubit emits (leaks)
/// - Stage 2 applies depolarizing to qubits whose partner emitted
///
/// # Example
///
/// ```
/// use pecos_neo::noise::composite::prelude::*;
///
/// // Two-stage emission with partner depolarizing
/// let noise = two_stage(
///     prob(0.01, sample_emission()),           // Stage 1: sample emission
///     when(partner_only_fired(), pauli(), nothing())  // Stage 2: partner depolarizing
/// );
/// ```
pub struct TwoStage<P1: Primitive, P2: Primitive> {
    stage1: P1,
    stage2: P2,
}

impl<P1: Primitive + Clone, P2: Primitive + Clone> Clone for TwoStage<P1, P2> {
    fn clone(&self) -> Self {
        Self {
            stage1: self.stage1.clone(),
            stage2: self.stage2.clone(),
        }
    }
}

impl<P1: Primitive, P2: Primitive> TwoStage<P1, P2> {
    /// Create a new two-stage primitive.
    #[must_use]
    pub fn new(stage1: P1, stage2: P2) -> Self {
        Self { stage1, stage2 }
    }

    /// Get a reference to stage 1.
    #[must_use]
    pub fn stage1(&self) -> &P1 {
        &self.stage1
    }

    /// Get a reference to stage 2.
    #[must_use]
    pub fn stage2(&self) -> &P2 {
        &self.stage2
    }
}

impl<P1: Primitive, P2: Primitive> Primitive for TwoStage<P1, P2> {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        // Fallback for single-qubit processing: run both stages sequentially
        let r1 = self.stage1.apply(qubit, ctx, rng);
        let r2 = self.stage2.apply(qubit, ctx, rng);
        r1.combine(r2)
    }

    fn describe(&self) -> String {
        format!(
            "two_stage({}, {})",
            self.stage1.describe(),
            self.stage2.describe()
        )
    }

    fn needs_two_pass(&self) -> bool {
        true
    }

    fn apply_stage1(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        self.stage1.apply(qubit, ctx, rng)
    }

    fn apply_stage2(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        self.stage2.apply(qubit, ctx, rng)
    }

    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(TwoStage {
            stage1: self.stage1.clone_box(),
            stage2: self.stage2.clone_box(),
        })
    }
}

impl Primitive for Box<dyn Primitive> {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        (**self).apply(qubit, ctx, rng)
    }

    fn describe(&self) -> String {
        (**self).describe()
    }

    fn describe_tree(&self) -> String {
        (**self).describe_tree()
    }

    fn describe_tree_with_prefix(&self, prefix: &str, is_last: bool) -> String {
        (**self).describe_tree_with_prefix(prefix, is_last)
    }

    fn needs_two_pass(&self) -> bool {
        (**self).needs_two_pass()
    }

    fn apply_stage1(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        (**self).apply_stage1(qubit, ctx, rng)
    }

    fn apply_stage2(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        (**self).apply_stage2(qubit, ctx, rng)
    }

    fn clone_box(&self) -> Box<dyn Primitive> {
        (**self).clone_box()
    }
}

// Implement Primitive for all GateActions
impl<A: GateAction + Clone + 'static> Primitive for A {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        GateAction::apply(self, qubit, ctx, rng)
    }

    fn describe(&self) -> String {
        self.name().to_string()
    }

    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}

/// Probability gate: with probability p, execute inner primitive.
///
/// This is the primary mechanism for early-exit optimization.
/// If the random check fails, no further evaluation happens.
pub struct Prob<P: Primitive> {
    probability: f64,
    inner: P,
}

impl<P: Primitive + Clone> Clone for Prob<P> {
    fn clone(&self) -> Self {
        Self {
            probability: self.probability,
            inner: self.inner.clone(),
        }
    }
}

impl<P: Primitive> Prob<P> {
    /// Create a new probability gate.
    ///
    /// # Arguments
    /// * `probability` - Probability of executing inner (0.0 to 1.0)
    /// * `inner` - Primitive to execute if probability check passes
    #[must_use]
    pub fn new(probability: f64, inner: P) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
            inner,
        }
    }

    /// Get the probability value.
    #[must_use]
    pub fn probability(&self) -> f64 {
        self.probability
    }

    /// Get a reference to the inner primitive.
    #[must_use]
    pub fn inner(&self) -> &P {
        &self.inner
    }
}

impl<P: Primitive> Primitive for Prob<P> {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        if rng.random::<f64>() < self.probability {
            self.inner.apply(qubit, ctx, rng)
        } else {
            CompositeResponse::None
        }
    }

    fn describe(&self) -> String {
        format!("prob({:.4}, {})", self.probability, self.inner.describe())
    }

    fn describe_tree(&self) -> String {
        format!(
            "Prob({:.4})\n└─ {}",
            self.probability,
            self.inner.describe_tree().replace('\n', "\n   ")
        )
    }

    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(Prob {
            probability: self.probability,
            inner: self.inner.clone_box(),
        })
    }
}

/// Dynamic probability gate: compute probability from gate context.
///
/// This primitive allows angle-dependent or gate-type-dependent error rates.
/// The probability function receives the current gate information and returns
/// a probability in [0.0, 1.0].
///
/// # Example
///
/// ```
/// use pecos_neo::noise::composite::prelude::*;
///
/// // Angle-dependent two-qubit noise
/// let tq_noise = prob_fn(
///     |gate| {
///         // Higher error rate for larger rotation angles
///         gate.and_then(|g| g.angle())
///             .map(|a| 0.01 * a.to_radians().abs())
///             .unwrap_or(0.01)
///     },
///     pauli(),
/// );
/// ```
pub struct ProbFn<F, P>
where
    F: Fn(Option<&crate::noise::GateInfo>) -> f64 + Send + Sync,
    P: Primitive,
{
    probability_fn: F,
    inner: P,
}

impl<F, P> Clone for ProbFn<F, P>
where
    F: Fn(Option<&crate::noise::GateInfo>) -> f64 + Send + Sync + Clone,
    P: Primitive + Clone,
{
    fn clone(&self) -> Self {
        Self {
            probability_fn: self.probability_fn.clone(),
            inner: self.inner.clone(),
        }
    }
}

impl<F, P> ProbFn<F, P>
where
    F: Fn(Option<&crate::noise::GateInfo>) -> f64 + Send + Sync,
    P: Primitive,
{
    /// Create a new dynamic probability gate.
    ///
    /// # Arguments
    /// * `probability_fn` - Function that computes probability from gate context
    /// * `inner` - Primitive to execute if probability check passes
    #[must_use]
    pub fn new(probability_fn: F, inner: P) -> Self {
        Self {
            probability_fn,
            inner,
        }
    }
}

impl<F, P> Primitive for ProbFn<F, P>
where
    F: Fn(Option<&crate::noise::GateInfo>) -> f64 + Send + Sync + Clone + 'static,
    P: Primitive,
{
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        let probability = (self.probability_fn)(ctx.current_gate()).clamp(0.0, 1.0);
        if rng.random::<f64>() < probability {
            self.inner.apply(qubit, ctx, rng)
        } else {
            CompositeResponse::None
        }
    }

    fn describe(&self) -> String {
        "prob_fn(...)".to_string()
    }

    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(ProbFn {
            probability_fn: self.probability_fn.clone(),
            inner: self.inner.clone_box(),
        })
    }
}

/// Linear time-dependent probability: p = rate * duration.
///
/// This models T1-like relaxation where error probability grows linearly with time.
/// The rate is specified per time unit (whatever units are used in the simulation).
///
/// # Example
///
/// ```
/// use pecos_neo::noise::composite::prelude::*;
///
/// // 0.001 error rate per time unit, applying Z error
/// let idle_noise = prob_linear(0.001, pauli());
/// ```
pub struct ProbLinear<P: Primitive> {
    rate_per_time_unit: f64,
    inner: P,
}

impl<P: Primitive + Clone> Clone for ProbLinear<P> {
    fn clone(&self) -> Self {
        Self {
            rate_per_time_unit: self.rate_per_time_unit,
            inner: self.inner.clone(),
        }
    }
}

impl<P: Primitive> ProbLinear<P> {
    /// Create a new linear time-dependent probability gate.
    ///
    /// # Arguments
    /// * `rate_per_time_unit` - Error rate per time unit
    /// * `inner` - Primitive to execute if probability check passes
    #[must_use]
    pub fn new(rate_per_time_unit: f64, inner: P) -> Self {
        Self {
            rate_per_time_unit: rate_per_time_unit.max(0.0),
            inner,
        }
    }
}

impl<P: Primitive> Primitive for ProbLinear<P> {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        let duration = ctx
            .current_idle()
            .map_or(0.0, crate::noise::IdleInfo::duration_f64);
        let probability = (self.rate_per_time_unit * duration).clamp(0.0, 1.0);
        if rng.random::<f64>() < probability {
            self.inner.apply(qubit, ctx, rng)
        } else {
            CompositeResponse::None
        }
    }

    fn describe(&self) -> String {
        format!("prob_linear({:.6})", self.rate_per_time_unit)
    }

    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(ProbLinear {
            rate_per_time_unit: self.rate_per_time_unit,
            inner: self.inner.clone_box(),
        })
    }
}

/// Quadratic time-dependent dephasing: p = sin(rate * duration)^2.
///
/// This models T2-like dephasing where the probability follows a sinusoidal pattern.
/// For incoherent dephasing (stochastic Z errors), this gives the probability of error.
///
/// For longer durations or higher rates, the probability saturates at 50% (average
/// of sin^2 over many cycles), which is the expected steady-state for random dephasing.
///
/// # Example
///
/// ```
/// use pecos_neo::noise::composite::prelude::*;
///
/// // Quadratic dephasing with 0.01 rate
/// let dephasing = prob_quadratic(0.01, inject_z());
/// ```
pub struct ProbQuadratic<P: Primitive> {
    rate_per_time_unit: f64,
    coherent_to_incoherent_factor: f64,
    inner: P,
}

impl<P: Primitive + Clone> Clone for ProbQuadratic<P> {
    fn clone(&self) -> Self {
        Self {
            rate_per_time_unit: self.rate_per_time_unit,
            coherent_to_incoherent_factor: self.coherent_to_incoherent_factor,
            inner: self.inner.clone(),
        }
    }
}

impl<P: Primitive> ProbQuadratic<P> {
    /// Create a new quadratic time-dependent probability gate.
    ///
    /// # Arguments
    /// * `rate_per_time_unit` - Dephasing rate per time unit
    /// * `inner` - Primitive to execute if probability check passes
    #[must_use]
    pub fn new(rate_per_time_unit: f64, inner: P) -> Self {
        Self {
            rate_per_time_unit: rate_per_time_unit.max(0.0),
            coherent_to_incoherent_factor: 1.0,
            inner,
        }
    }

    /// Set the coherent-to-incoherent scaling factor.
    ///
    /// This factor adjusts the effective rate when modeling coherent dephasing
    /// using stochastic errors. Default is 1.0 (no adjustment).
    #[must_use]
    pub fn with_factor(mut self, factor: f64) -> Self {
        self.coherent_to_incoherent_factor = factor;
        self
    }
}

impl<P: Primitive> Primitive for ProbQuadratic<P> {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        let duration = ctx
            .current_idle()
            .map_or(0.0, crate::noise::IdleInfo::duration_f64);
        let angle = self.rate_per_time_unit * self.coherent_to_incoherent_factor * duration;
        // sin^2(angle) gives probability of dephasing
        let probability = angle.sin().powi(2).clamp(0.0, 1.0);
        if rng.random::<f64>() < probability {
            self.inner.apply(qubit, ctx, rng)
        } else {
            CompositeResponse::None
        }
    }

    fn describe(&self) -> String {
        format!("prob_quadratic({:.6})", self.rate_per_time_unit)
    }

    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(ProbQuadratic {
            rate_per_time_unit: self.rate_per_time_unit,
            coherent_to_incoherent_factor: self.coherent_to_incoherent_factor,
            inner: self.inner.clone_box(),
        })
    }
}

/// Conditional: if condition is true, execute `then_branch`, else `else_branch`.
pub struct When<C: Condition, T: Primitive, E: Primitive> {
    condition: C,
    then_branch: T,
    else_branch: E,
}

impl<C: Condition + Clone, T: Primitive + Clone, E: Primitive + Clone> Clone for When<C, T, E> {
    fn clone(&self) -> Self {
        Self {
            condition: self.condition.clone(),
            then_branch: self.then_branch.clone(),
            else_branch: self.else_branch.clone(),
        }
    }
}

impl<C: Condition, T: Primitive, E: Primitive> When<C, T, E> {
    /// Create a new conditional primitive.
    #[must_use]
    pub fn new(condition: C, then_branch: T, else_branch: E) -> Self {
        Self {
            condition,
            then_branch,
            else_branch,
        }
    }
}

impl<C: Condition + Clone + 'static, T: Primitive, E: Primitive> Primitive for When<C, T, E> {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        if self.condition.evaluate(qubit, ctx) {
            self.then_branch.apply(qubit, ctx, rng)
        } else {
            self.else_branch.apply(qubit, ctx, rng)
        }
    }

    fn describe(&self) -> String {
        format!(
            "when({}, {}, {})",
            self.condition.name(),
            self.then_branch.describe(),
            self.else_branch.describe()
        )
    }

    fn describe_tree(&self) -> String {
        format!(
            "When({})\n├─ then: {}\n└─ else: {}",
            self.condition.name(),
            self.then_branch.describe_tree().replace('\n', "\n│  "),
            self.else_branch.describe_tree().replace('\n', "\n   ")
        )
    }

    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(When {
            condition: self.condition.clone(),
            then_branch: self.then_branch.clone_box(),
            else_branch: self.else_branch.clone_box(),
        })
    }
}

/// Weighted sample: choose one branch based on weights.
///
/// Weights are normalized internally, so they don't need to sum to 1.0.
pub struct Sample<P: Primitive> {
    branches: Vec<(f64, P)>,
    cumulative_weights: Vec<f64>,
}

impl<P: Primitive + Clone> Clone for Sample<P> {
    fn clone(&self) -> Self {
        Self {
            branches: self.branches.clone(),
            cumulative_weights: self.cumulative_weights.clone(),
        }
    }
}

impl<P: Primitive> Sample<P> {
    /// Create a new weighted sample primitive.
    ///
    /// Weights are normalized to sum to 1.0.
    #[must_use]
    pub fn new(branches: Vec<(f64, P)>) -> Self {
        let total: f64 = branches.iter().map(|(w, _)| w).sum();
        let normalized: Vec<f64> = branches.iter().map(|(w, _)| w / total).collect();

        let mut cumulative = Vec::with_capacity(normalized.len());
        let mut sum = 0.0;
        for w in normalized {
            sum += w;
            cumulative.push(sum);
        }

        Self {
            branches,
            cumulative_weights: cumulative,
        }
    }
}

impl<P: Primitive> Primitive for Sample<P> {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        if self.branches.is_empty() {
            return CompositeResponse::None;
        }

        let r: f64 = rng.random();

        for (i, &threshold) in self.cumulative_weights.iter().enumerate() {
            if r < threshold {
                return self.branches[i].1.apply(qubit, ctx, rng);
            }
        }

        // Fallback to last branch (floating-point rounding guard)
        self.branches
            .last()
            .expect("Sample must have at least one branch")
            .1
            .apply(qubit, ctx, rng)
    }

    fn describe(&self) -> String {
        let branches: Vec<String> = self
            .branches
            .iter()
            .map(|(w, p)| format!("({w:.2}, {})", p.describe()))
            .collect();
        format!("sample([{}])", branches.join(", "))
    }

    fn describe_tree(&self) -> String {
        let mut result = String::from("Sample\n");
        for (i, (weight, primitive)) in self.branches.iter().enumerate() {
            let is_last = i == self.branches.len() - 1;
            let connector = if is_last { "└─" } else { "├─" };
            let prefix = if is_last { "   " } else { "│  " };
            let _ = writeln!(
                result,
                "{} ({:.2}) {}",
                connector,
                weight,
                primitive
                    .describe_tree()
                    .replace('\n', &format!("\n{prefix}"))
            );
        }
        result.trim_end().to_string()
    }

    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(BoxSample::new(
            self.branches
                .iter()
                .map(|(w, p)| (*w, p.clone_box()))
                .collect(),
        ))
    }
}

/// Sequential: execute all primitives in order, combine responses.
pub struct Seq<P: Primitive> {
    primitives: Vec<P>,
}

impl<P: Primitive + Clone> Clone for Seq<P> {
    fn clone(&self) -> Self {
        Self {
            primitives: self.primitives.clone(),
        }
    }
}

impl<P: Primitive> Seq<P> {
    /// Create a new sequential primitive.
    #[must_use]
    pub fn new(primitives: Vec<P>) -> Self {
        Self { primitives }
    }
}

impl<P: Primitive> Primitive for Seq<P> {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        let mut combined = CompositeResponse::None;

        for prim in &self.primitives {
            let response = prim.apply(qubit, ctx, rng);

            // If we hit a SkipGate, stop processing and return it
            if response.skips_gate() {
                return combined.combine(response);
            }

            combined = combined.combine(response);
        }

        combined
    }

    fn describe(&self) -> String {
        let items: Vec<String> = self.primitives.iter().map(Primitive::describe).collect();
        format!("seq([{}])", items.join(", "))
    }

    fn describe_tree(&self) -> String {
        let mut result = String::from("Seq\n");
        for (i, primitive) in self.primitives.iter().enumerate() {
            let is_last = i == self.primitives.len() - 1;
            let connector = if is_last { "└─" } else { "├─" };
            let prefix = if is_last { "   " } else { "│  " };
            let _ = writeln!(
                result,
                "{} {}",
                connector,
                primitive
                    .describe_tree()
                    .replace('\n', &format!("\n{prefix}"))
            );
        }
        result.trim_end().to_string()
    }

    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(BoxSeq::new(
            self.primitives.iter().map(Primitive::clone_box).collect(),
        ))
    }
}

/// Sequential execution of heterogeneous primitives using trait objects.
///
/// Unlike `Seq<P>` which requires all primitives to be the same type,
/// `BoxSeq` allows different primitive types by boxing them.
pub struct BoxSeq {
    primitives: Vec<Box<dyn Primitive>>,
}

impl Clone for BoxSeq {
    fn clone(&self) -> Self {
        Self {
            primitives: self.primitives.iter().map(Primitive::clone_box).collect(),
        }
    }
}

impl BoxSeq {
    /// Create a new boxed sequential primitive.
    #[must_use]
    pub fn new(primitives: Vec<Box<dyn Primitive>>) -> Self {
        Self { primitives }
    }
}

impl Primitive for BoxSeq {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        let mut combined = CompositeResponse::None;

        for prim in &self.primitives {
            let response = prim.apply(qubit, ctx, rng);

            // If we hit a SkipGate, stop processing and return it
            if response.skips_gate() {
                return combined.combine(response);
            }

            combined = combined.combine(response);
        }

        combined
    }

    fn describe(&self) -> String {
        let items: Vec<String> = self.primitives.iter().map(Primitive::describe).collect();
        format!("seq([{}])", items.join(", "))
    }

    fn describe_tree(&self) -> String {
        let mut result = String::from("Seq\n");
        for (i, primitive) in self.primitives.iter().enumerate() {
            let is_last = i == self.primitives.len() - 1;
            let connector = if is_last { "└─" } else { "├─" };
            let prefix = if is_last { "   " } else { "│  " };
            let _ = writeln!(
                result,
                "{} {}",
                connector,
                primitive
                    .describe_tree()
                    .replace('\n', &format!("\n{prefix}"))
            );
        }
        result.trim_end().to_string()
    }

    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}

/// Early exit: if condition is true, return `SkipGate` response.
pub struct SkipIf<C: Condition> {
    condition: C,
}

impl<C: Condition + Clone> Clone for SkipIf<C> {
    fn clone(&self) -> Self {
        Self {
            condition: self.condition.clone(),
        }
    }
}

impl<C: Condition> SkipIf<C> {
    /// Create a new skip-if primitive.
    #[must_use]
    pub fn new(condition: C) -> Self {
        Self { condition }
    }
}

impl<C: Condition + Clone + 'static> Primitive for SkipIf<C> {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        if self.condition.evaluate(qubit, ctx) {
            CompositeResponse::SkipGate
        } else {
            CompositeResponse::None
        }
    }

    fn describe(&self) -> String {
        format!("skip_if({})", self.condition.name())
    }

    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}

/// Create a boxed sequence from heterogeneous primitives.
///
/// This macro boxes each primitive and creates a `BoxSeq`.
///
/// # Example
///
/// ```
/// use pecos_neo::noise::composite::prelude::*;
///
/// let noise = seq![
///     skip_if_leaked(),
///     prob(0.01, pauli()),
/// ];
/// ```
#[macro_export]
macro_rules! seq {
    ($($prim:expr),* $(,)?) => {
        $crate::noise::composite::BoxSeq::new(vec![
            $(Box::new($prim) as Box<dyn $crate::noise::composite::Primitive>),*
        ])
    };
}

/// Create a boxed sample from heterogeneous primitives with weights.
///
/// This macro boxes each primitive and creates a `BoxSample`.
///
/// # Example
///
/// ```
/// use pecos_neo::noise::composite::prelude::*;
///
/// let noise = sample![
///     (0.25, leak()),
///     (0.75, pauli()),
/// ];
/// ```
#[macro_export]
macro_rules! sample {
    ($(($weight:expr, $prim:expr)),* $(,)?) => {
        $crate::noise::composite::BoxSample::new(vec![
            $(($weight, Box::new($prim) as Box<dyn $crate::noise::composite::Primitive>)),*
        ])
    };
}

/// Weighted sample of heterogeneous primitives using trait objects.
///
/// Unlike `Sample<P>` which requires all primitives to be the same type,
/// `BoxSample` allows different primitive types by boxing them.
pub struct BoxSample {
    branches: Vec<(f64, Box<dyn Primitive>)>,
    cumulative_weights: Vec<f64>,
}

impl Clone for BoxSample {
    fn clone(&self) -> Self {
        Self {
            branches: self
                .branches
                .iter()
                .map(|(w, p)| (*w, p.clone_box()))
                .collect(),
            cumulative_weights: self.cumulative_weights.clone(),
        }
    }
}

impl BoxSample {
    /// Create a new boxed weighted sample primitive.
    ///
    /// Weights are normalized to sum to 1.0.
    #[must_use]
    pub fn new(branches: Vec<(f64, Box<dyn Primitive>)>) -> Self {
        let total: f64 = branches.iter().map(|(w, _)| w).sum();
        let normalized: Vec<f64> = branches.iter().map(|(w, _)| w / total).collect();

        let mut cumulative = Vec::with_capacity(normalized.len());
        let mut sum = 0.0;
        for w in normalized {
            sum += w;
            cumulative.push(sum);
        }

        Self {
            branches,
            cumulative_weights: cumulative,
        }
    }
}

impl Primitive for BoxSample {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        if self.branches.is_empty() {
            return CompositeResponse::None;
        }

        let r: f64 = rng.random();

        for (i, &threshold) in self.cumulative_weights.iter().enumerate() {
            if r < threshold {
                return self.branches[i].1.apply(qubit, ctx, rng);
            }
        }

        // Fallback to last branch (floating-point rounding guard)
        self.branches
            .last()
            .expect("BoxSample must have at least one branch")
            .1
            .apply(qubit, ctx, rng)
    }

    fn describe(&self) -> String {
        let branches: Vec<String> = self
            .branches
            .iter()
            .map(|(w, p)| format!("({w:.2}, {})", p.describe()))
            .collect();
        format!("sample([{}])", branches.join(", "))
    }

    fn describe_tree(&self) -> String {
        let mut result = String::from("Sample\n");
        for (i, (weight, primitive)) in self.branches.iter().enumerate() {
            let is_last = i == self.branches.len() - 1;
            let connector = if is_last { "└─" } else { "├─" };
            let prefix = if is_last { "   " } else { "│  " };
            let _ = writeln!(
                result,
                "{} ({:.2}) {}",
                connector,
                weight,
                primitive
                    .describe_tree()
                    .replace('\n', &format!("\n{prefix}"))
            );
        }
        result.trim_end().to_string()
    }

    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}

/// Convenience functions for creating primitives.
pub mod primitives {
    use super::{
        Condition, Primitive, Prob, ProbFn, ProbLinear, ProbQuadratic, Sample, Seq, SkipIf,
        TwoStage, When,
    };
    use crate::noise::GateInfo;
    use crate::noise::composite::action::Nothing;
    use crate::noise::composite::condition::{Always, GateTypeIs, Leaked, NotLeaked, OutcomeIs};

    /// Probability gate.
    #[must_use]
    pub fn prob<P: Primitive>(probability: f64, inner: P) -> Prob<P> {
        Prob::new(probability, inner)
    }

    /// Dynamic probability gate based on gate context.
    ///
    /// The probability function receives the current gate information
    /// (if available) and returns a probability in [0.0, 1.0].
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::prelude::*;
    ///
    /// // Angle-dependent error: p = 0.01 * |angle|
    /// let noise = prob_fn(
    ///     |gate| {
    ///         gate.and_then(|g| g.angle())
    ///             .map(|a| 0.01 * a.to_radians().abs())
    ///             .unwrap_or(0.01)
    ///     },
    ///     pauli(),
    /// );
    /// ```
    #[must_use]
    pub fn prob_fn<F, P>(probability_fn: F, inner: P) -> ProbFn<F, P>
    where
        F: Fn(Option<&GateInfo>) -> f64 + Send + Sync,
        P: Primitive,
    {
        ProbFn::new(probability_fn, inner)
    }

    /// Linear time-dependent probability: p = rate * duration.
    ///
    /// This models T1-like relaxation where error probability grows linearly with time.
    /// Requires idle context to be set (via `NoiseContext::set_current_idle`).
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::prelude::*;
    ///
    /// // T1 relaxation: 0.001 error per time unit
    /// let t1_noise = prob_linear(0.001, pauli());
    /// ```
    #[must_use]
    pub fn prob_linear<P: Primitive>(rate_per_time_unit: f64, inner: P) -> ProbLinear<P> {
        ProbLinear::new(rate_per_time_unit, inner)
    }

    /// Quadratic time-dependent probability: p = sin(rate * duration)^2.
    ///
    /// This models T2-like dephasing. Requires idle context to be set.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::prelude::*;
    ///
    /// // T2 dephasing
    /// let t2_noise = prob_quadratic(0.01, inject_z());
    /// ```
    #[must_use]
    pub fn prob_quadratic<P: Primitive>(rate_per_time_unit: f64, inner: P) -> ProbQuadratic<P> {
        ProbQuadratic::new(rate_per_time_unit, inner)
    }

    /// Conditional (with else branch).
    #[must_use]
    pub fn when<C: Condition, T: Primitive, E: Primitive>(
        condition: C,
        then_branch: T,
        else_branch: E,
    ) -> When<C, T, E> {
        When::new(condition, then_branch, else_branch)
    }

    /// Conditional when leaked (with else branch).
    #[must_use]
    pub fn when_leaked<T: Primitive, E: Primitive>(
        then_branch: T,
        else_branch: E,
    ) -> When<Leaked, T, E> {
        When::new(Leaked, then_branch, else_branch)
    }

    /// Conditional when not leaked (with else branch).
    #[must_use]
    pub fn when_not_leaked<T: Primitive, E: Primitive>(
        then_branch: T,
        else_branch: E,
    ) -> When<NotLeaked, T, E> {
        When::new(NotLeaked, then_branch, else_branch)
    }

    /// Conditional when partner is leaked in a two-qubit gate.
    ///
    /// This applies the `then_branch` to THIS qubit when the OTHER qubit
    /// in a two-qubit gate is leaked (and this qubit is not).
    ///
    /// # Example
    ///
    /// ```
    /// # use pecos_neo::noise::composite::prelude::*;
    /// // Partner depolarizing: apply random Pauli when partner is leaked
    /// when_partner_leaked(pauli(), nothing());
    /// ```
    #[must_use]
    pub fn when_partner_leaked<T: Primitive, E: Primitive>(
        then_branch: T,
        else_branch: E,
    ) -> When<crate::noise::composite::PartnerLeaked, T, E> {
        When::new(
            crate::noise::composite::PartnerLeaked,
            then_branch,
            else_branch,
        )
    }

    /// Conditional (then only, else is nothing).
    #[must_use]
    pub fn if_then<C: Condition, T: Primitive>(
        condition: C,
        then_branch: T,
    ) -> When<C, T, Nothing> {
        When::new(condition, then_branch, Nothing)
    }

    /// Always execute (useful for seq).
    #[must_use]
    pub fn always<P: Primitive>(inner: P) -> When<Always, P, Nothing> {
        When::new(Always, inner, Nothing)
    }

    /// Weighted sample (homogeneous types).
    ///
    /// For heterogeneous types, use the `sample!` macro.
    #[must_use]
    pub fn sample_uniform<P: Primitive>(branches: Vec<(f64, P)>) -> Sample<P> {
        Sample::new(branches)
    }

    /// Sequential execution (homogeneous types).
    ///
    /// For heterogeneous types, use the `seq!` macro.
    #[must_use]
    pub fn seq_uniform<P: Primitive>(primitives: Vec<P>) -> Seq<P> {
        Seq::new(primitives)
    }

    /// Skip gate if condition is true.
    #[must_use]
    pub fn skip_if<C: Condition>(condition: C) -> SkipIf<C> {
        SkipIf::new(condition)
    }

    /// Skip gate if qubit is leaked.
    #[must_use]
    pub fn skip_if_leaked() -> SkipIf<Leaked> {
        SkipIf::new(Leaked)
    }

    // ========================================================================
    // Outcome-conditional primitives (for measurement noise)
    // ========================================================================

    /// Execute action only if the current measurement outcome matches the value.
    ///
    /// This is a convenience for `if_then(outcome_is(value), action)`.
    ///
    /// # Example
    ///
    /// ```
    /// # use pecos_neo::noise::composite::prelude::*;
    /// // Asymmetric measurement noise
    /// let meas_noise = seq![
    ///     on_outcome(false, prob(0.02, flip_outcome())),  // 2% error on 0
    ///     on_outcome(true, prob(0.05, flip_outcome())),   // 5% error on 1
    /// ];
    /// ```
    #[must_use]
    pub fn on_outcome<T: Primitive>(value: bool, action: T) -> When<OutcomeIs, T, Nothing> {
        When::new(OutcomeIs::new(value), action, Nothing)
    }

    /// Execute action only if the current measurement outcome is 0.
    #[must_use]
    pub fn on_zero<T: Primitive>(action: T) -> When<OutcomeIs, T, Nothing> {
        on_outcome(false, action)
    }

    /// Execute action only if the current measurement outcome is 1.
    #[must_use]
    pub fn on_one<T: Primitive>(action: T) -> When<OutcomeIs, T, Nothing> {
        on_outcome(true, action)
    }

    /// Execute action only if the current gate type matches.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::prelude::*;
    /// use pecos_neo::command::GateType;
    ///
    /// // Special handling for MeasureLeaked gates
    /// let noise = seq![
    ///     on_gate_type(GateType::MeasureLeaked, when_leaked(leaked_measurement(), nothing())),
    /// ];
    /// ```
    #[must_use]
    pub fn on_gate_type<T: Primitive>(
        gate_type: crate::command::GateType,
        action: T,
    ) -> When<GateTypeIs, T, Nothing> {
        When::new(GateTypeIs::new(gate_type), action, Nothing)
    }

    // ========================================================================
    // Angle-scaled probability helpers (for two-qubit gates)
    // ========================================================================

    /// Probability scaled by gate angle: p = `base_prob` * |angle/π|^power.
    ///
    /// This models angle-dependent error rates for parameterized gates like
    /// RZZ, RXX, RYY, CRZ. The scaling factor is |θ/π|^power, which:
    /// - Is 0 when angle = 0 (no error for identity-like gates)
    /// - Is 1 when angle = π (full error for maximally-entangling gates)
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::prelude::*;
    ///
    /// // Linear angle scaling: p = 0.01 * |angle/π|
    /// let noise = prob_angle_scaled(0.01, 1.0, pauli());
    ///
    /// // Quadratic angle scaling: p = 0.01 * |angle/π|^2
    /// let noise = prob_angle_scaled(0.01, 2.0, pauli());
    /// ```
    #[must_use]
    pub fn prob_angle_scaled<P: Primitive>(
        base_probability: f64,
        power: f64,
        inner: P,
    ) -> ProbFn<impl Fn(Option<&GateInfo>) -> f64 + Send + Sync, P> {
        ProbFn::new(
            move |gate: Option<&GateInfo>| {
                let scale = gate
                    .and_then(crate::noise::context::GateInfo::angle)
                    .map_or(1.0, |a| {
                        (a.to_radians().abs() / std::f64::consts::PI).powf(power)
                    });
                (base_probability * scale).min(1.0)
            },
            inner,
        )
    }

    /// Probability with linear angle scaling: p = `base_prob` * |angle/π|.
    ///
    /// Convenience wrapper for `prob_angle_scaled(base_prob, 1.0, inner)`.
    #[must_use]
    pub fn prob_angle_linear<P: Primitive>(
        base_probability: f64,
        inner: P,
    ) -> ProbFn<impl Fn(Option<&GateInfo>) -> f64 + Send + Sync, P> {
        prob_angle_scaled(base_probability, 1.0, inner)
    }

    /// Probability with quadratic angle scaling: p = `base_prob` * |angle/π|^2.
    ///
    /// Convenience wrapper for `prob_angle_scaled(base_prob, 2.0, inner)`.
    #[must_use]
    pub fn prob_angle_quadratic<P: Primitive>(
        base_probability: f64,
        inner: P,
    ) -> ProbFn<impl Fn(Option<&GateInfo>) -> f64 + Send + Sync, P> {
        prob_angle_scaled(base_probability, 2.0, inner)
    }

    // ========================================================================
    // Two-Stage Primitives (for correlated multi-qubit effects)
    // ========================================================================

    /// Create a two-stage primitive for proper cross-qubit condition handling.
    ///
    /// Two-stage primitives are processed in two passes:
    /// 1. **Stage 1**: Run `stage1` on ALL qubits (typically sampling/firing)
    /// 2. **Stage 2**: Run `stage2` on ALL qubits (typically cross-effects)
    ///
    /// This enables proper handling of cross-qubit conditions in multi-qubit gates,
    /// such as "if partner emitted, depolarize me".
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::prelude::*;
    ///
    /// // Two-stage emission with partner depolarizing
    /// let noise = two_stage(
    ///     prob(0.01, sample_emission()),           // Stage 1: sample emission
    ///     when(partner_only_fired(), pauli(), nothing())  // Stage 2: partner depolarizing
    /// );
    /// ```
    ///
    /// # How It Works
    ///
    /// For a two-qubit gate with qubits A and B:
    /// - Stage 1: Sample emission for A, then sample emission for B
    /// - Stage 2: Check conditions for A (can see B's result), then for B (can see A's result)
    ///
    /// Without two-stage processing, sequential processing of A then B would mean
    /// when processing A we can't know if B will emit yet.
    #[must_use]
    pub fn two_stage<P1: Primitive, P2: Primitive>(stage1: P1, stage2: P2) -> TwoStage<P1, P2> {
        TwoStage::new(stage1, stage2)
    }
}

#[cfg(test)]
mod tests {
    use super::primitives::*;
    use super::*;
    use crate::command::GateType;
    use crate::noise::composite::action::actions::*;
    use crate::noise::composite::condition::Leaked;
    use pecos_core::Angle64;

    fn make_test_context() -> NoiseContext {
        NoiseContext::new()
    }

    #[test]
    fn test_prob_passes() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // Probability 1.0 should always pass
        let prim = prob(1.0, leak());
        let response = prim.apply(QubitId(0), &mut ctx, &mut rng);

        assert!(response.causes_leak());
    }

    #[test]
    fn test_prob_fails() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // Probability 0.0 should never pass
        let prim = prob(0.0, leak());
        let response = prim.apply(QubitId(0), &mut ctx, &mut rng);

        assert!(response.is_none());
    }

    #[test]
    fn test_prob_statistical() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        let prim = prob(0.3, leak());
        let mut leak_count = 0;

        for _ in 0..1000 {
            ctx.mark_unleaked(QubitId(0)); // Reset for each trial
            let response = prim.apply(QubitId(0), &mut ctx, &mut rng);
            if response.causes_leak() {
                leak_count += 1;
            }
        }

        // Should be roughly 30%
        let rate = f64::from(leak_count) / 1000.0;
        assert!(
            (rate - 0.3).abs() < 0.05,
            "Expected ~30% leak rate, got {rate}"
        );
    }

    #[test]
    fn test_when_then_branch() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        ctx.mark_leaked(QubitId(0));

        let prim = when(Leaked, leak(), nothing());
        let response = prim.apply(QubitId(0), &mut ctx, &mut rng);

        assert!(response.causes_leak());
    }

    #[test]
    fn test_when_else_branch() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // Qubit is NOT leaked
        let prim = when_leaked(leak(), skip_gate());
        let response = prim.apply(QubitId(0), &mut ctx, &mut rng);

        assert!(response.skips_gate());
        assert!(!response.causes_leak());
    }

    #[test]
    fn test_sample_distribution() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // 70% inject X, 30% inject Z (homogeneous - both Inject type)
        let prim = sample_uniform(vec![(0.7, inject_x()), (0.3, inject_z())]);

        let mut x_count = 0;
        let mut z_count = 0;

        for _ in 0..1000 {
            let response = prim.apply(QubitId(0), &mut ctx, &mut rng);
            let gates = response.collect_gates();
            assert_eq!(gates.len(), 1);

            match gates[0].gate_type {
                crate::command::GateType::X => x_count += 1,
                crate::command::GateType::Z => z_count += 1,
                _ => panic!("Unexpected gate"),
            }
        }

        let x_rate = f64::from(x_count) / 1000.0;
        let z_rate = f64::from(z_count) / 1000.0;

        assert!((x_rate - 0.7).abs() < 0.05, "Expected ~70% X, got {x_rate}");
        assert!((z_rate - 0.3).abs() < 0.05, "Expected ~30% Z, got {z_rate}");
    }

    #[test]
    fn test_seq_combines_responses() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // Homogeneous sequence - both Inject type
        let prim = seq_uniform(vec![inject_x(), inject_z()]);
        let response = prim.apply(QubitId(0), &mut ctx, &mut rng);

        let gates = response.collect_gates();
        assert_eq!(gates.len(), 2);
    }

    #[test]
    fn test_seq_stops_on_skip() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        ctx.mark_leaked(QubitId(0));

        // skip_if_leaked should stop the sequence (heterogeneous - use macro)
        let prim = seq![skip_if_leaked(), inject_x()];
        let response = prim.apply(QubitId(0), &mut ctx, &mut rng);

        assert!(response.skips_gate());
        // X gate should NOT be injected because we stopped early
        let gates = response.collect_gates();
        assert!(gates.is_empty());
    }

    #[test]
    fn test_complex_tree() {
        // Build something like SQ noise:
        // skip_if(leaked)
        // prob(0.5,
        //   when(leaked,
        //     seep(),
        //     pauli()
        //   )
        // )

        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // Heterogeneous sequence - use macro
        let noise = seq![skip_if_leaked(), prob(0.5, when_leaked(seep(), pauli())),];

        // Not leaked: should sometimes apply pauli
        let mut pauli_count = 0;
        for _ in 0..1000 {
            let response = noise.apply(QubitId(0), &mut ctx, &mut rng);
            if !response.collect_gates().is_empty() {
                pauli_count += 1;
            }
        }

        // Should be roughly 50%
        let rate = f64::from(pauli_count) / 1000.0;
        assert!(
            (rate - 0.5).abs() < 0.05,
            "Expected ~50% pauli rate, got {rate}"
        );
    }

    #[test]
    fn test_prob_fn_with_gate_context() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // Dynamic probability based on gate type
        // CX gets 100% error, H gets 0% error
        let noise = prob_fn(
            |gate| {
                gate.map_or(0.0, |g| {
                    if g.gate_type == GateType::CX {
                        1.0
                    } else {
                        0.0
                    }
                })
            },
            inject_x(),
        );

        // Set gate context to H (should not trigger)
        ctx.set_current_gate(GateType::H, &[], 1);
        let response = noise.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(response.is_none());
        ctx.clear_current_gate();

        // Set gate context to CX (should always trigger)
        ctx.set_current_gate(GateType::CX, &[], 2);
        let response = noise.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(!response.is_none());
        let gates = response.collect_gates();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::X);
        ctx.clear_current_gate();
    }

    #[test]
    fn test_prob_fn_angle_dependent() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // Angle-dependent probability: p = angle / pi
        // So half turn (pi) = 100% error, quarter turn (pi/2) = 50% error
        let noise = prob_fn(
            |gate| {
                gate.and_then(crate::noise::context::GateInfo::angle)
                    .map_or(0.0, |a| a.to_radians() / std::f64::consts::PI)
            },
            inject_z(),
        );

        // Half turn angle should always trigger
        ctx.set_current_gate(GateType::RZ, &[Angle64::HALF_TURN], 1);
        let mut triggered = 0;
        for _ in 0..100 {
            let response = noise.apply(QubitId(0), &mut ctx, &mut rng);
            if !response.is_none() {
                triggered += 1;
            }
        }
        // Should be ~100%
        assert!(
            triggered > 95,
            "Expected ~100% trigger rate for half turn, got {triggered}%"
        );
        ctx.clear_current_gate();

        // Quarter turn should trigger ~50%
        ctx.set_current_gate(GateType::RZ, &[Angle64::QUARTER_TURN], 1);
        triggered = 0;
        for _ in 0..1000 {
            let response = noise.apply(QubitId(0), &mut ctx, &mut rng);
            if !response.is_none() {
                triggered += 1;
            }
        }
        let rate = f64::from(triggered) / 1000.0;
        assert!(
            (rate - 0.5).abs() < 0.1,
            "Expected ~50% trigger rate for quarter turn, got {rate}"
        );
        ctx.clear_current_gate();
    }

    #[test]
    fn test_prob_fn_no_context_fallback() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // Default to 50% when no gate context
        let noise = prob_fn(
            |gate| {
                if gate.is_some() { 1.0 } else { 0.5 }
            },
            inject_x(),
        );

        // No gate context set - should use fallback (50%)
        let mut triggered = 0;
        for _ in 0..1000 {
            let response = noise.apply(QubitId(0), &mut ctx, &mut rng);
            if !response.is_none() {
                triggered += 1;
            }
        }
        let rate = f64::from(triggered) / 1000.0;
        assert!(
            (rate - 0.5).abs() < 0.1,
            "Expected ~50% trigger rate with no context, got {rate}"
        );
    }

    // ========================================================================
    // Time-dependent probability tests (for idle noise)
    // ========================================================================

    #[test]
    fn test_prob_linear_with_duration() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // Linear rate: 0.1 per time unit
        let noise = prob_linear(0.1, inject_z());

        // Set idle duration to 10 units -> p = 0.1 * 10 = 1.0 (100%)
        ctx.set_current_idle(pecos_core::TimeUnits::new(10));
        let mut triggered = 0;
        for _ in 0..100 {
            let response = noise.apply(QubitId(0), &mut ctx, &mut rng);
            if !response.is_none() {
                triggered += 1;
            }
        }
        assert!(
            triggered > 95,
            "Expected ~100% trigger rate for duration=10, got {triggered}%"
        );
        ctx.clear_current_idle();

        // Set idle duration to 5 units -> p = 0.1 * 5 = 0.5 (50%)
        ctx.set_current_idle(pecos_core::TimeUnits::new(5));
        triggered = 0;
        for _ in 0..1000 {
            let response = noise.apply(QubitId(0), &mut ctx, &mut rng);
            if !response.is_none() {
                triggered += 1;
            }
        }
        let rate = f64::from(triggered) / 1000.0;
        assert!(
            (rate - 0.5).abs() < 0.1,
            "Expected ~50% trigger rate for duration=5, got {rate}"
        );
        ctx.clear_current_idle();
    }

    #[test]
    fn test_prob_linear_no_context() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // Without idle context, duration is 0 -> p = 0
        let noise = prob_linear(0.1, inject_z());

        let mut triggered = 0;
        for _ in 0..100 {
            let response = noise.apply(QubitId(0), &mut ctx, &mut rng);
            if !response.is_none() {
                triggered += 1;
            }
        }
        assert_eq!(
            triggered, 0,
            "Expected 0% trigger rate without idle context"
        );
    }

    #[test]
    fn test_prob_quadratic_dephasing() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // Quadratic rate for T2 dephasing: p = sin(rate * duration)^2
        // At rate = pi/2 and duration = 1, angle = pi/2, p = sin(pi/2)^2 = 1.0
        let rate = std::f64::consts::FRAC_PI_2;
        let noise = prob_quadratic(rate, inject_z());

        // Duration = 1 -> angle = pi/2 -> p = 100%
        ctx.set_current_idle(pecos_core::TimeUnits::new(1));
        let mut triggered = 0;
        for _ in 0..100 {
            let response = noise.apply(QubitId(0), &mut ctx, &mut rng);
            if !response.is_none() {
                triggered += 1;
            }
        }
        assert!(triggered > 95, "Expected ~100% trigger rate for angle=pi/2");
        ctx.clear_current_idle();

        // For 50% probability, we need sin(angle)^2 = 0.5, so angle = pi/4
        // Rate = pi/4, duration = 1 -> angle = pi/4 -> p = sin(pi/4)^2 = 0.5
        let rate_half = std::f64::consts::FRAC_PI_4;
        let noise_half = prob_quadratic(rate_half, inject_z());
        ctx.set_current_idle(pecos_core::TimeUnits::new(1));
        triggered = 0;
        for _ in 0..1000 {
            let response = noise_half.apply(QubitId(0), &mut ctx, &mut rng);
            if !response.is_none() {
                triggered += 1;
            }
        }
        let measured_rate = f64::from(triggered) / 1000.0;
        assert!(
            (measured_rate - 0.5).abs() < 0.1,
            "Expected ~50% trigger rate for angle=pi/4, got {measured_rate}"
        );
        ctx.clear_current_idle();
    }

    #[test]
    fn test_prob_quadratic_with_factor() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // Base rate pi/4, with factor 2.0 -> effective rate = pi/2
        // At duration = 1, angle = pi/2 -> p = 100%
        let noise = prob_quadratic(std::f64::consts::FRAC_PI_4, inject_z()).with_factor(2.0);

        ctx.set_current_idle(pecos_core::TimeUnits::new(1));
        let mut triggered = 0;
        for _ in 0..100 {
            let response = noise.apply(QubitId(0), &mut ctx, &mut rng);
            if !response.is_none() {
                triggered += 1;
            }
        }
        assert!(
            triggered > 95,
            "Expected ~100% trigger rate with factor=2.0"
        );
        ctx.clear_current_idle();
    }
}
