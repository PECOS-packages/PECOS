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

//! Actions for noise decision trees.
//!
//! Actions are the leaf nodes of noise decision trees - they produce
//! concrete `CompositeResponse` values that specify what noise to apply.

use super::response::CompositeResponse;
use crate::command::{GateCommand, GateType};
use crate::noise::NoiseContext;
use pecos_core::QubitId;
use pecos_random::PecosRng;
use rand::RngExt;
use smallvec::smallvec;

/// A terminal action that produces a noise response.
///
/// Actions are the leaf nodes of noise decision trees. Unlike primitives
/// (which can contain other primitives), actions directly produce responses.
pub trait GateAction: Send + Sync {
    /// Apply this action for a specific qubit.
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse;

    /// Human-readable name for visualization.
    fn name(&self) -> &'static str;
}

/// No-op action - does nothing.
#[derive(Debug, Clone, Copy, Default)]
pub struct Nothing;

impl GateAction for Nothing {
    fn apply(
        &self,
        _qubit: QubitId,
        _ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        CompositeResponse::None
    }

    fn name(&self) -> &'static str {
        "nothing"
    }
}

/// Skip/remove the current gate for this qubit.
#[derive(Debug, Clone, Copy, Default)]
pub struct SkipGate;

impl GateAction for SkipGate {
    fn apply(
        &self,
        _qubit: QubitId,
        _ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        CompositeResponse::SkipGate
    }

    fn name(&self) -> &'static str {
        "skip_gate"
    }
}

/// Mark qubit as leaked.
#[derive(Debug, Clone, Copy, Default)]
pub struct Leak;

impl GateAction for Leak {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        ctx.mark_leaked(qubit);
        CompositeResponse::Leak
    }

    fn name(&self) -> &'static str {
        "leak"
    }
}

/// Mark qubit as unleaked (seeped back to computational basis).
#[derive(Debug, Clone, Copy, Default)]
pub struct Unleak;

impl GateAction for Unleak {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        ctx.mark_unleaked(qubit);
        CompositeResponse::Unleak
    }

    fn name(&self) -> &'static str {
        "unleak"
    }
}

/// Inject a specific gate.
#[derive(Debug, Clone)]
pub struct Inject {
    gate_type: GateType,
}

impl Inject {
    /// Create an action that injects a specific gate type.
    #[must_use]
    pub fn new(gate_type: GateType) -> Self {
        Self { gate_type }
    }

    /// Inject an X gate.
    #[must_use]
    pub fn x() -> Self {
        Self::new(GateType::X)
    }

    /// Inject a Y gate.
    #[must_use]
    pub fn y() -> Self {
        Self::new(GateType::Y)
    }

    /// Inject a Z gate.
    #[must_use]
    pub fn z() -> Self {
        Self::new(GateType::Z)
    }
}

impl GateAction for Inject {
    fn apply(
        &self,
        qubit: QubitId,
        _ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        let cmd = GateCommand {
            gate_type: self.gate_type,
            qubits: smallvec![qubit],
            angles: smallvec![],
        };
        CompositeResponse::InjectGates(vec![cmd])
    }

    fn name(&self) -> &'static str {
        "inject"
    }
}

/// Pauli weights for random Pauli sampling.
#[derive(Debug, Clone, Copy)]
pub struct PauliWeights {
    /// Weight for X error.
    pub x: f64,
    /// Weight for Y error.
    pub y: f64,
    /// Weight for Z error.
    pub z: f64,
}

impl Default for PauliWeights {
    fn default() -> Self {
        Self::uniform()
    }
}

impl PauliWeights {
    /// Uniform weights (1/3 each).
    #[must_use]
    pub fn uniform() -> Self {
        Self {
            x: 1.0,
            y: 1.0,
            z: 1.0,
        }
    }

    /// Custom weights.
    #[must_use]
    pub fn custom(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// Z-only (dephasing).
    #[must_use]
    pub fn z_only() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    }

    /// Normalize weights to sum to 1.
    #[must_use]
    pub fn normalized(&self) -> Self {
        let total = self.x + self.y + self.z;
        if total <= 0.0 {
            return Self::uniform();
        }
        Self {
            x: self.x / total,
            y: self.y / total,
            z: self.z / total,
        }
    }
}

/// Apply a random Pauli gate based on weights.
#[derive(Debug, Clone)]
pub struct Pauli {
    weights: PauliWeights,
}

impl Pauli {
    /// Create a Pauli action with specified weights.
    #[must_use]
    pub fn new(weights: PauliWeights) -> Self {
        Self {
            weights: weights.normalized(),
        }
    }

    /// Uniform depolarizing (equal X, Y, Z probability).
    #[must_use]
    pub fn uniform() -> Self {
        Self::new(PauliWeights::uniform())
    }

    /// Z-only (dephasing).
    #[must_use]
    pub fn dephasing() -> Self {
        Self::new(PauliWeights::z_only())
    }
}

impl Default for Pauli {
    fn default() -> Self {
        Self::uniform()
    }
}

impl GateAction for Pauli {
    fn apply(
        &self,
        qubit: QubitId,
        _ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        let r: f64 = rng.random();

        let gate_type = if r < self.weights.x {
            GateType::X
        } else if r < self.weights.x + self.weights.y {
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

    fn name(&self) -> &'static str {
        "pauli"
    }
}

/// Seepage: unleak qubit and apply random Pauli.
///
/// This models a leaked qubit spontaneously returning to the computational
/// basis in a random state.
#[derive(Debug, Clone, Default)]
pub struct Seep {
    pauli_weights: PauliWeights,
}

impl Seep {
    /// Create a seep action with uniform Pauli distribution.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a seep action with custom Pauli weights.
    #[must_use]
    pub fn with_weights(weights: PauliWeights) -> Self {
        Self {
            pauli_weights: weights.normalized(),
        }
    }
}

impl GateAction for Seep {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        // Unleak the qubit
        ctx.mark_unleaked(qubit);

        // Apply random Pauli (or nothing - 25% chance for I)
        let r: f64 = rng.random();
        if r < 0.25 {
            // Identity - just unleak
            return CompositeResponse::Unleak;
        }

        // Scale to remaining 75%
        let r_scaled = (r - 0.25) / 0.75;
        let weights = self.pauli_weights.normalized();

        let gate_type = if r_scaled < weights.x {
            GateType::X
        } else if r_scaled < weights.x + weights.y {
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

    fn name(&self) -> &'static str {
        "seep"
    }
}

// ============================================================================
// Two-Qubit Actions
// ============================================================================

/// Apply a correlated two-qubit Pauli error.
///
/// This samples from all 15 non-identity two-qubit Pauli operators
/// (XI, YI, ZI, IX, IY, IZ, XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ).
///
/// Unlike `Pauli` which samples independently per qubit, this action
/// properly samples correlated errors like XX or YZ.
///
/// Note: This action requires the second qubit ID from context.
/// Use with `CompositeChannel` for two-qubit gates where both qubits
/// are processed together.
#[derive(Debug, Clone)]
pub struct TwoQubitPauli {
    weights: crate::noise::TwoQubitPauliWeights,
}

impl TwoQubitPauli {
    /// Create with uniform weights (1/15 each).
    #[must_use]
    pub fn uniform() -> Self {
        Self {
            weights: crate::noise::TwoQubitPauliWeights::uniform(),
        }
    }

    /// Create with custom weights.
    #[must_use]
    pub fn with_weights(weights: crate::noise::TwoQubitPauliWeights) -> Self {
        Self { weights }
    }

    /// Create with ZZ-biased weights.
    #[must_use]
    pub fn zz_biased(zz_weight: f64) -> Self {
        Self {
            weights: crate::noise::TwoQubitPauliWeights::zz_biased(zz_weight),
        }
    }
}

impl Default for TwoQubitPauli {
    fn default() -> Self {
        Self::uniform()
    }
}

impl GateAction for TwoQubitPauli {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        let qubit_index = ctx.current_qubit_index();

        // Get or sample the two-qubit Pauli index
        let idx = if qubit_index == 0 {
            // First qubit: sample and store
            let r: f64 = rng.random();
            let sampled_idx = self.weights.sample(r);
            ctx.set_sampled_correlation(sampled_idx);
            sampled_idx
        } else {
            // Second qubit: retrieve stored value
            ctx.sampled_correlation().unwrap_or_else(|| {
                // Fallback: sample again (shouldn't happen in normal use)
                let r: f64 = rng.random();
                self.weights.sample(r)
            })
        };

        let (first_pauli, second_pauli) = crate::noise::TwoQubitPauliWeights::get_paulis(idx);

        // Get the Pauli for this qubit based on index
        let my_pauli = if qubit_index == 0 {
            first_pauli
        } else {
            second_pauli
        };

        // Only inject if not identity
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

    fn name(&self) -> &'static str {
        "two_qubit_pauli"
    }
}

// ============================================================================
// Emission Actions
// ============================================================================

/// Apply an emission error (Pauli or leakage).
///
/// This samples from the emission distribution which can include
/// X, Y, Z Pauli errors and/or leakage. This matches the
/// `SingleQubitEmissionWeights` structure from `GeneralNoiseModel`.
#[derive(Debug, Clone)]
pub struct Emission {
    weights: crate::noise::SingleQubitEmissionWeights,
}

impl Emission {
    /// Create with uniform weights (equal X, Y, Z, Leak).
    #[must_use]
    pub fn uniform() -> Self {
        Self {
            weights: crate::noise::SingleQubitEmissionWeights::uniform(),
        }
    }

    /// Create with custom weights.
    #[must_use]
    pub fn with_weights(weights: crate::noise::SingleQubitEmissionWeights) -> Self {
        Self { weights }
    }

    /// Create with Pauli-only weights (no leakage).
    #[must_use]
    pub fn pauli_only() -> Self {
        Self {
            weights: crate::noise::SingleQubitEmissionWeights::uniform(),
        }
    }

    /// Create with leakage probability.
    ///
    /// Remaining probability is split equally among X, Y, Z.
    #[must_use]
    pub fn with_leakage(leak_prob: f64) -> Self {
        let pauli_prob = (1.0 - leak_prob) / 3.0;
        Self {
            weights: crate::noise::SingleQubitEmissionWeights::custom(
                pauli_prob, pauli_prob, pauli_prob, leak_prob,
            ),
        }
    }
}

impl Default for Emission {
    fn default() -> Self {
        Self::uniform()
    }
}

impl GateAction for Emission {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        let r: f64 = rng.random();
        let result = self.weights.sample(r);

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

    fn name(&self) -> &'static str {
        "emission"
    }
}

/// Apply a correlated two-qubit emission error (Pauli and/or leakage).
///
/// This samples from all 24 two-qubit emission operators which include
/// Pauli errors (X, Y, Z, I) and leakage (L) on each qubit.
///
/// Uses context correlation to ensure both qubits receive correlated errors.
#[derive(Debug, Clone)]
pub struct TwoQubitEmission {
    weights: crate::noise::TwoQubitEmissionWeights,
}

impl TwoQubitEmission {
    /// Create with uniform Pauli weights (no leakage).
    #[must_use]
    pub fn uniform_pauli() -> Self {
        Self {
            weights: crate::noise::TwoQubitEmissionWeights::uniform_pauli(),
        }
    }

    /// Create with uniform weights including leakage.
    #[must_use]
    pub fn uniform_with_leakage() -> Self {
        Self {
            weights: crate::noise::TwoQubitEmissionWeights::uniform_with_leakage(),
        }
    }

    /// Create with custom weights.
    #[must_use]
    pub fn with_weights(weights: crate::noise::TwoQubitEmissionWeights) -> Self {
        Self { weights }
    }
}

impl Default for TwoQubitEmission {
    fn default() -> Self {
        Self::uniform_pauli()
    }
}

impl GateAction for TwoQubitEmission {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        let qubit_index = ctx.current_qubit_index();

        // Get or sample the two-qubit emission index
        let idx = if qubit_index == 0 {
            // First qubit: sample and store
            let r: f64 = rng.random();
            let sampled_idx = self.weights.sample(r);
            ctx.set_sampled_correlation(sampled_idx);
            sampled_idx
        } else {
            // Second qubit: retrieve stored value
            ctx.sampled_correlation().unwrap_or_else(|| {
                // Fallback: sample again (shouldn't happen in normal use)
                let r: f64 = rng.random();
                self.weights.sample(r)
            })
        };

        let result = crate::noise::TwoQubitEmissionWeights::get_result(idx);

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

    fn name(&self) -> &'static str {
        "two_qubit_emission"
    }
}

// ============================================================================
// Sample Emission Action (for two-stage composite)
// ============================================================================

/// Sample whether an emission event occurs and store the result.
///
/// This action is used in stage 1 of two-stage composite processing:
/// - Stage 1: Sample emission for each qubit, store "fired" flag
/// - Stage 2: Apply effects based on cross-conditions (e.g., partner depolarizing)
///
/// The action stores the result in `ctx.fired_flags` so that stage 2 conditions
/// like `IFired`, `PartnerFired`, and `PartnerFiredAndIDidnt` can evaluate it.
///
/// # Example
///
/// ```
/// # use pecos_neo::noise::composite::prelude::*;
/// // Stage 1: Sample emission
/// let stage1 = sample_emission();  // Stores fired flag, returns Leak if fired
///
/// // Stage 2: Apply partner depolarizing based on fired flags
/// let stage2 = when(partner_only_fired(), pauli(), nothing());
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct SampleEmission;

impl SampleEmission {
    /// Create a new sample emission action.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl GateAction for SampleEmission {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        // This action is used inside prob(), so if we got here, the event fired
        let index = ctx.current_qubit_index();
        ctx.set_fired(index, true);

        // Mark as leaked
        ctx.mark_leaked(qubit);
        CompositeResponse::Leak
    }

    fn name(&self) -> &'static str {
        "sample_emission"
    }
}

/// Sample emission with probability (for stage 1 of two-stage composite).
///
/// Unlike `SampleEmission` which assumes it's inside a `prob()`, this action
/// handles the probability sampling itself.
#[derive(Debug, Clone, Copy)]
pub struct SampleEmissionWithProb {
    prob: f64,
}

impl SampleEmissionWithProb {
    /// Create with given emission probability.
    #[must_use]
    pub fn new(prob: f64) -> Self {
        Self {
            prob: prob.clamp(0.0, 1.0),
        }
    }
}

impl GateAction for SampleEmissionWithProb {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        let index = ctx.current_qubit_index();
        let fired = rng.random::<f64>() < self.prob;

        ctx.set_fired(index, fired);

        if fired {
            ctx.mark_leaked(qubit);
            CompositeResponse::Leak
        } else {
            CompositeResponse::None
        }
    }

    fn name(&self) -> &'static str {
        "sample_emission_with_prob"
    }
}

// ============================================================================
// Two-Qubit Emission with Partner Depolarize
// ============================================================================

/// Two-qubit emission that handles partner depolarizing correctly.
///
/// This models the physical process for two-qubit gates:
/// 1. Each qubit independently samples whether it spontaneously emits (leaks)
/// 2. If one emitted and the other didn't, the non-emitter gets depolarized
///
/// The action coordinates between qubits using the context's correlation storage:
/// - Qubit 0: Sample emission, store result, mark as leaked if emitted
/// - Qubit 1: Sample emission, check cross-conditions, apply partner depolarizing
///
/// # Example
///
/// ```
/// use pecos_neo::noise::composite::prelude::*;
///
/// // Two-qubit gate with emission probability and partner depolarizing
/// let tq_noise = prob(0.01, two_qubit_emission_with_partner_depolarize());
/// ```
#[derive(Debug, Clone, Default)]
pub struct TwoQubitEmissionWithPartnerDepolarize {
    /// Pauli weights for partner depolarizing (when one leaks, other gets Pauli)
    partner_pauli_weights: PauliWeights,
}

impl TwoQubitEmissionWithPartnerDepolarize {
    /// Create with uniform partner depolarizing (1/3 X, Y, Z).
    #[must_use]
    pub fn new() -> Self {
        Self {
            partner_pauli_weights: PauliWeights::uniform(),
        }
    }

    /// Create with custom partner depolarizing weights.
    #[must_use]
    pub fn with_partner_weights(weights: PauliWeights) -> Self {
        Self {
            partner_pauli_weights: weights.normalized(),
        }
    }

    /// Sample a random Pauli gate type based on weights.
    fn sample_pauli(&self, rng: &mut PecosRng) -> GateType {
        let r: f64 = rng.random();
        let weights = &self.partner_pauli_weights;
        if r < weights.x {
            GateType::X
        } else if r < weights.x + weights.y {
            GateType::Y
        } else {
            GateType::Z
        }
    }
}

// Encoding for stored emission state:
// 0 = qubit 0 did NOT emit
// 1 = qubit 0 DID emit
const EMIT_NO: usize = 0;
const EMIT_YES: usize = 1;

impl GateAction for TwoQubitEmissionWithPartnerDepolarize {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        let qubit_index = ctx.current_qubit_index();

        if qubit_index == 0 {
            // First qubit: this IS an emission event (we're inside prob())
            // Mark as leaked and store that we emitted
            ctx.mark_leaked(qubit);
            ctx.set_sampled_correlation(EMIT_YES);
            CompositeResponse::Leak
        } else {
            // Second qubit: this IS an emission event for us too
            // Check what happened to qubit 0
            let qubit0_emitted = ctx.sampled_correlation() == Some(EMIT_YES);

            // Mark ourselves as leaked
            ctx.mark_leaked(qubit);

            // If qubit 0 did NOT emit, it needs to be depolarized
            // (we emitted, partner didn't → depolarize partner)
            if !qubit0_emitted && let Some(other) = ctx.other_qubit() {
                let pauli = self.sample_pauli(rng);
                let cmd = GateCommand {
                    gate_type: pauli,
                    qubits: smallvec![other],
                    angles: smallvec![],
                };
                return CompositeResponse::Leak.combine(CompositeResponse::InjectGates(vec![cmd]));
            }

            CompositeResponse::Leak
        }
    }

    fn name(&self) -> &'static str {
        "two_qubit_emission_with_partner_depolarize"
    }
}

/// Independent emission sampling for two-qubit gates with partner depolarizing.
///
/// Unlike `TwoQubitEmissionWithPartnerDepolarize` which assumes the action is
/// only called when emission occurs, this action samples emission independently
/// for each qubit and handles all cross-conditions.
///
/// Use this when wrapping with `prob()` isn't sufficient (e.g., different
/// emission probabilities per qubit).
#[derive(Debug, Clone)]
pub struct IndependentEmissionWithPartnerDepolarize {
    /// Probability of emission for each qubit
    emission_prob: f64,
    /// Pauli weights for partner depolarizing
    partner_pauli_weights: PauliWeights,
}

impl IndependentEmissionWithPartnerDepolarize {
    /// Create with given emission probability.
    #[must_use]
    pub fn new(emission_prob: f64) -> Self {
        Self {
            emission_prob: emission_prob.clamp(0.0, 1.0),
            partner_pauli_weights: PauliWeights::uniform(),
        }
    }

    /// Create with custom partner depolarizing weights.
    #[must_use]
    pub fn with_partner_weights(mut self, weights: PauliWeights) -> Self {
        self.partner_pauli_weights = weights.normalized();
        self
    }

    fn sample_pauli(&self, rng: &mut PecosRng) -> GateType {
        let r: f64 = rng.random();
        let weights = &self.partner_pauli_weights;
        if r < weights.x {
            GateType::X
        } else if r < weights.x + weights.y {
            GateType::Y
        } else {
            GateType::Z
        }
    }
}

impl GateAction for IndependentEmissionWithPartnerDepolarize {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        let qubit_index = ctx.current_qubit_index();

        if qubit_index == 0 {
            // First qubit: sample emission, store result
            let emitted = rng.random::<f64>() < self.emission_prob;
            ctx.set_sampled_correlation(if emitted { EMIT_YES } else { EMIT_NO });

            if emitted {
                ctx.mark_leaked(qubit);
                CompositeResponse::Leak
            } else {
                CompositeResponse::None
            }
        } else {
            // Second qubit: sample emission, check cross-conditions
            let q0_emitted = ctx.sampled_correlation() == Some(EMIT_YES);
            let q1_emitted = rng.random::<f64>() < self.emission_prob;

            let mut response = CompositeResponse::None;

            // Handle our own emission
            if q1_emitted {
                ctx.mark_leaked(qubit);
                response = CompositeResponse::Leak;
            }

            // Handle partner depolarizing based on cross-conditions
            if let Some(other) = ctx.other_qubit() {
                // If q0 emitted but q1 didn't → depolarize q1 (this qubit)
                if q0_emitted && !q1_emitted {
                    let pauli = self.sample_pauli(rng);
                    let cmd = GateCommand {
                        gate_type: pauli,
                        qubits: smallvec![qubit],
                        angles: smallvec![],
                    };
                    response = response.combine(CompositeResponse::InjectGates(vec![cmd]));
                }

                // If q1 emitted but q0 didn't → depolarize q0 (other qubit)
                if q1_emitted && !q0_emitted {
                    let pauli = self.sample_pauli(rng);
                    let cmd = GateCommand {
                        gate_type: pauli,
                        qubits: smallvec![other],
                        angles: smallvec![],
                    };
                    response = response.combine(CompositeResponse::InjectGates(vec![cmd]));
                }
            }

            response
        }
    }

    fn name(&self) -> &'static str {
        "independent_emission_with_partner_depolarize"
    }
}

// ============================================================================
// Partner Depolarize Action (for two-qubit gates with leakage)
// ============================================================================

/// Apply depolarizing noise to the OTHER qubit in a two-qubit gate.
///
/// This action is used when one qubit of a two-qubit gate is leaked:
/// the leaked qubit receives no error, but its partner receives a random
/// Pauli error (depolarizing). This models the physical effect where
/// a leaked qubit cannot participate in the intended gate, causing
/// the other qubit to effectively receive noise.
///
/// The action looks up the "other qubit" from the context and applies
/// a random Pauli (X, Y, or Z with equal probability) to it.
///
/// # Usage
///
/// This is typically used in a two-qubit gate channel with a condition:
/// ```
/// # use pecos_neo::noise::composite::prelude::*;
/// // If this qubit is leaked, depolarize its partner
/// let noise = when_leaked(partner_depolarize(), nothing());
/// ```
#[derive(Debug, Clone, Default)]
pub struct PartnerDepolarize {
    weights: PauliWeights,
}

impl PartnerDepolarize {
    /// Create with uniform Pauli weights (1/3 each for X, Y, Z).
    #[must_use]
    pub fn uniform() -> Self {
        Self {
            weights: PauliWeights::uniform(),
        }
    }

    /// Create with custom Pauli weights.
    #[must_use]
    pub fn with_weights(weights: PauliWeights) -> Self {
        Self {
            weights: weights.normalized(),
        }
    }
}

impl GateAction for PartnerDepolarize {
    fn apply(
        &self,
        _qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        // Get the other qubit in this two-qubit gate
        let Some(other) = ctx.other_qubit() else {
            // Not a two-qubit gate or can't find partner
            return CompositeResponse::None;
        };

        // Don't depolarize if partner is also leaked
        if ctx.is_leaked(other) {
            return CompositeResponse::None;
        }

        // Sample random Pauli
        let r: f64 = rng.random();
        let gate_type = if r < self.weights.x {
            GateType::X
        } else if r < self.weights.x + self.weights.y {
            GateType::Y
        } else {
            GateType::Z
        };

        let cmd = GateCommand {
            gate_type,
            qubits: smallvec![other],
            angles: smallvec![],
        };
        CompositeResponse::InjectGates(vec![cmd])
    }

    fn name(&self) -> &'static str {
        "partner_depolarize"
    }
}

// ============================================================================
// Coherent Dephasing Actions
// ============================================================================

/// Inject an RZ gate for coherent dephasing.
///
/// This injects an actual RZ rotation gate rather than a stochastic Z error.
/// The angle is computed from the idle duration and rate.
///
/// For coherent dephasing: RZ(rate * duration)
#[derive(Debug, Clone)]
pub struct InjectCoherentRZ {
    rate: f64,
}

impl InjectCoherentRZ {
    /// Create with a given dephasing rate.
    #[must_use]
    pub fn new(rate: f64) -> Self {
        Self { rate }
    }
}

impl GateAction for InjectCoherentRZ {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        let duration = ctx
            .current_idle()
            .map_or(0.0, crate::noise::IdleInfo::duration_f64);

        let angle = self.rate * duration;

        // Only inject if angle is non-trivial
        if angle.abs() < 1e-10 {
            return CompositeResponse::None;
        }

        let cmd = GateCommand {
            gate_type: GateType::RZ,
            qubits: smallvec![qubit],
            angles: smallvec![pecos_core::Angle64::from_radians(angle)],
        };
        CompositeResponse::InjectGates(vec![cmd])
    }

    fn name(&self) -> &'static str {
        "coherent_rz"
    }
}

// ============================================================================
// Amplitude Damping (T1 Relaxation)
// ============================================================================

/// Amplitude damping: asymmetric T1 relaxation where |1⟩ → |0⟩.
///
/// Unlike symmetric Pauli X which flips both |0⟩ ↔ |1⟩, amplitude damping
/// only decays |1⟩ → |0⟩. This models energy relaxation (T1 decay) where
/// the excited state loses energy to the environment.
///
/// When triggered, this action:
/// - Projects the qubit toward |0⟩ by applying a "relaxation" operation
/// - In a Pauli frame simulation, this is approximated by probabilistic X
///   only when the qubit is in |1⟩ state
///
/// For proper simulation, the state must be tracked. In stabilizer simulation,
/// this is typically approximated by random Pauli with bias toward Z.
#[derive(Debug, Clone, Copy, Default)]
pub struct AmplitudeDamping;

impl AmplitudeDamping {
    /// Create a new amplitude damping action.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl GateAction for AmplitudeDamping {
    fn apply(
        &self,
        qubit: QubitId,
        _ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        // In stabilizer simulation, amplitude damping is approximated as:
        // - 50% chance of Z error (phase flip, preserves |0⟩, flips phase of |1⟩)
        // - 50% chance of X error (bit flip, approximates decay)
        // This is a common approximation for T1 in Pauli frame simulations.
        let gate_type = if rng.random::<bool>() {
            GateType::Z
        } else {
            GateType::X
        };

        let cmd = GateCommand {
            gate_type,
            qubits: smallvec![qubit],
            angles: smallvec![],
        };
        CompositeResponse::InjectGates(vec![cmd])
    }

    fn name(&self) -> &'static str {
        "amplitude_damping"
    }
}

/// Biased amplitude damping with configurable X vs Z ratio.
///
/// Allows tuning the approximation of T1 decay in Pauli frame simulation.
/// - Higher X weight: more bit-flip like (aggressive relaxation approximation)
/// - Higher Z weight: more phase-flip like (conservative approximation)
#[derive(Debug, Clone, Copy)]
pub struct BiasedAmplitudeDamping {
    /// Probability of X error (vs Z error)
    x_probability: f64,
}

impl BiasedAmplitudeDamping {
    /// Create with given X probability (remainder is Z).
    #[must_use]
    pub fn new(x_probability: f64) -> Self {
        Self {
            x_probability: x_probability.clamp(0.0, 1.0),
        }
    }

    /// Standard T1 approximation (50% X, 50% Z).
    #[must_use]
    pub fn standard() -> Self {
        Self::new(0.5)
    }

    /// X-biased (more aggressive relaxation).
    #[must_use]
    pub fn x_biased() -> Self {
        Self::new(0.75)
    }

    /// Z-biased (more conservative, mostly dephasing).
    #[must_use]
    pub fn z_biased() -> Self {
        Self::new(0.25)
    }
}

impl Default for BiasedAmplitudeDamping {
    fn default() -> Self {
        Self::standard()
    }
}

impl GateAction for BiasedAmplitudeDamping {
    fn apply(
        &self,
        qubit: QubitId,
        _ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        let gate_type = if rng.random::<f64>() < self.x_probability {
            GateType::X
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

    fn name(&self) -> &'static str {
        "biased_amplitude_damping"
    }
}

// ============================================================================
// Coherent Rotation Errors
// ============================================================================

/// Inject an arbitrary coherent rotation error.
///
/// This injects an actual rotation gate (RX, RY, or RZ) with a fixed angle,
/// modeling systematic calibration errors or coherent noise.
#[derive(Debug, Clone)]
pub struct CoherentRotation {
    gate_type: GateType,
    angle: f64,
}

impl CoherentRotation {
    /// Create a coherent rotation error.
    #[must_use]
    pub fn new(gate_type: GateType, angle: f64) -> Self {
        Self { gate_type, angle }
    }

    /// RX rotation error.
    #[must_use]
    pub fn rx(angle: f64) -> Self {
        Self::new(GateType::RX, angle)
    }

    /// RY rotation error.
    #[must_use]
    pub fn ry(angle: f64) -> Self {
        Self::new(GateType::RY, angle)
    }

    /// RZ rotation error (phase error).
    #[must_use]
    pub fn rz(angle: f64) -> Self {
        Self::new(GateType::RZ, angle)
    }
}

impl GateAction for CoherentRotation {
    fn apply(
        &self,
        qubit: QubitId,
        _ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        if self.angle.abs() < 1e-10 {
            return CompositeResponse::None;
        }

        let cmd = GateCommand {
            gate_type: self.gate_type,
            qubits: smallvec![qubit],
            angles: smallvec![pecos_core::Angle64::from_radians(self.angle)],
        };
        CompositeResponse::InjectGates(vec![cmd])
    }

    fn name(&self) -> &'static str {
        "coherent_rotation"
    }
}

/// Over-rotation error: adds a fraction of the gate's angle as extra rotation.
///
/// Models systematic calibration errors where gates rotate slightly more
/// or less than intended. The error angle is computed as:
/// `error_angle = gate_angle * over_rotation_fraction`
///
/// # Example
///
/// ```
/// use pecos_neo::noise::composite::prelude::*;
/// use pecos_neo::command::GateType;
///
/// // 1% over-rotation on all gates
/// let error = over_rotation(GateType::RZ, 0.01);
/// ```
#[derive(Debug, Clone)]
pub struct OverRotation {
    gate_type: GateType,
    fraction: f64,
}

impl OverRotation {
    /// Create an over-rotation error.
    ///
    /// - `fraction > 0`: over-rotation (gate rotates too much)
    /// - `fraction < 0`: under-rotation (gate rotates too little)
    #[must_use]
    pub fn new(gate_type: GateType, fraction: f64) -> Self {
        Self {
            gate_type,
            fraction,
        }
    }

    /// RZ over-rotation.
    #[must_use]
    pub fn rz(fraction: f64) -> Self {
        Self::new(GateType::RZ, fraction)
    }
}

impl GateAction for OverRotation {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        // Get the gate's angle from context
        #[allow(clippy::redundant_closure)]
        let base_angle = ctx
            .current_gate()
            .and_then(super::super::context::GateInfo::angle)
            .map_or(0.0, |a| a.to_radians());

        let error_angle = base_angle * self.fraction;

        if error_angle.abs() < 1e-10 {
            return CompositeResponse::None;
        }

        let cmd = GateCommand {
            gate_type: self.gate_type,
            qubits: smallvec![qubit],
            angles: smallvec![pecos_core::Angle64::from_radians(error_angle)],
        };
        CompositeResponse::InjectGates(vec![cmd])
    }

    fn name(&self) -> &'static str {
        "over_rotation"
    }
}

// ============================================================================
// Correlated Phase Errors (ZZ Dephasing)
// ============================================================================

/// ZZ dephasing: correlated phase error on two qubits.
///
/// This models residual ZZ interaction during two-qubit gates or idle periods.
/// Applies RZZ(angle) which adds correlated phase based on both qubit states.
///
/// In stabilizer simulation, this is approximated by correlated Z errors.
#[derive(Debug, Clone)]
pub struct ZZDephasing {
    angle: f64,
}

impl ZZDephasing {
    /// Create ZZ dephasing with fixed angle.
    #[must_use]
    pub fn new(angle: f64) -> Self {
        Self { angle }
    }

    /// Create ZZ dephasing scaled by idle duration.
    ///
    /// The actual angle will be `rate * duration` from the idle context.
    #[must_use]
    pub fn from_rate(rate: f64) -> ZZDephasingRate {
        ZZDephasingRate { rate }
    }
}

impl GateAction for ZZDephasing {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        if self.angle.abs() < 1e-10 {
            return CompositeResponse::None;
        }

        // Get the other qubit for RZZ
        let Some(other) = ctx.other_qubit() else {
            // Not a two-qubit context, apply Z to this qubit only
            let cmd = GateCommand {
                gate_type: GateType::RZ,
                qubits: smallvec![qubit],
                angles: smallvec![pecos_core::Angle64::from_radians(self.angle)],
            };
            return CompositeResponse::InjectGates(vec![cmd]);
        };

        // Apply RZZ to both qubits (only on first qubit to avoid double-counting)
        if ctx.current_qubit_index() == 0 {
            let cmd = GateCommand {
                gate_type: GateType::RZZ,
                qubits: smallvec![qubit, other],
                angles: smallvec![pecos_core::Angle64::from_radians(self.angle)],
            };
            CompositeResponse::InjectGates(vec![cmd])
        } else {
            // Second qubit: already handled by first
            CompositeResponse::None
        }
    }

    fn name(&self) -> &'static str {
        "zz_dephasing"
    }
}

/// ZZ dephasing with rate (angle = rate * duration).
#[derive(Debug, Clone)]
pub struct ZZDephasingRate {
    rate: f64,
}

impl GateAction for ZZDephasingRate {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        let duration = ctx
            .current_idle()
            .map_or(0.0, crate::noise::IdleInfo::duration_f64);

        let angle = self.rate * duration;

        if angle.abs() < 1e-10 {
            return CompositeResponse::None;
        }

        // Get the other qubit for RZZ
        let Some(other) = ctx.other_qubit() else {
            let cmd = GateCommand {
                gate_type: GateType::RZ,
                qubits: smallvec![qubit],
                angles: smallvec![pecos_core::Angle64::from_radians(angle)],
            };
            return CompositeResponse::InjectGates(vec![cmd]);
        };

        if ctx.current_qubit_index() == 0 {
            let cmd = GateCommand {
                gate_type: GateType::RZZ,
                qubits: smallvec![qubit, other],
                angles: smallvec![pecos_core::Angle64::from_radians(angle)],
            };
            CompositeResponse::InjectGates(vec![cmd])
        } else {
            CompositeResponse::None
        }
    }

    fn name(&self) -> &'static str {
        "zz_dephasing_rate"
    }
}

// ============================================================================
// Preparation Errors
// ============================================================================

/// Preparation error: qubit prepared in wrong state.
///
/// Models errors during state preparation where the qubit ends up
/// in the wrong computational basis state.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrepFlip;

impl PrepFlip {
    /// Create a preparation flip error (applies X).
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl GateAction for PrepFlip {
    fn apply(
        &self,
        qubit: QubitId,
        _ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        let cmd = GateCommand {
            gate_type: GateType::X,
            qubits: smallvec![qubit],
            angles: smallvec![],
        };
        CompositeResponse::InjectGates(vec![cmd])
    }

    fn name(&self) -> &'static str {
        "prep_flip"
    }
}

/// Preparation phase error: qubit prepared with wrong phase.
///
/// Applies Z error after preparation, modeling phase coherence issues.
#[derive(Debug, Clone, Copy, Default)]
pub struct PrepPhase;

impl PrepPhase {
    /// Create a preparation phase error (applies Z).
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl GateAction for PrepPhase {
    fn apply(
        &self,
        qubit: QubitId,
        _ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        let cmd = GateCommand {
            gate_type: GateType::Z,
            qubits: smallvec![qubit],
            angles: smallvec![],
        };
        CompositeResponse::InjectGates(vec![cmd])
    }

    fn name(&self) -> &'static str {
        "prep_phase"
    }
}

// ============================================================================
// Erasure/Heralded Errors
// ============================================================================

/// Erasure error: heralded loss of quantum information.
///
/// Unlike leakage (which may go undetected), erasure is a detectable
/// error where we know the qubit has been lost. This is modeled by:
/// 1. Marking the qubit as erased (similar to leaked but flagged)
/// 2. Returning a response that signals erasure occurred
///
/// Erasure errors are useful for:
/// - Modeling photon loss in optical systems
/// - Modeling detectable atom loss in neutral atom systems
/// - Erasure-based error correction schemes
#[derive(Debug, Clone, Copy, Default)]
pub struct Erasure;

impl Erasure {
    /// Create an erasure error action.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl GateAction for Erasure {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        // Mark as leaked (erasure is a form of leakage)
        ctx.mark_leaked(qubit);
        // Return erasure response (currently same as Leak, but semantically distinct)
        // In the future, could have a distinct Erasure variant in CompositeResponse
        CompositeResponse::Leak
    }

    fn name(&self) -> &'static str {
        "erasure"
    }
}

/// Erasure with replacement: erase and reinitialize qubit.
///
/// Models systems where erased qubits can be replaced with fresh ones
/// in a known state (e.g., atom reloading).
#[derive(Debug, Clone, Copy, Default)]
pub struct ErasureWithReplacement;

impl ErasureWithReplacement {
    /// Create erasure with replacement.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl GateAction for ErasureWithReplacement {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        // Unleak (reset to computational basis) with random Pauli
        ctx.mark_unleaked(qubit);

        // Apply random Pauli to model unknown replacement state
        let gate_type = match rng.random::<u8>() % 4 {
            0 => return CompositeResponse::None, // Identity
            1 => GateType::X,
            2 => GateType::Y,
            _ => GateType::Z,
        };

        let cmd = GateCommand {
            gate_type,
            qubits: smallvec![qubit],
            angles: smallvec![],
        };
        CompositeResponse::InjectGates(vec![cmd])
    }

    fn name(&self) -> &'static str {
        "erasure_with_replacement"
    }
}

// ============================================================================
// Crosstalk Actions
// ============================================================================

/// Apply state-dependent crosstalk effects.
///
/// This action uses `CrosstalkTransitions` to model what happens to a qubit
/// when it experiences crosstalk. The effect depends on the qubit's current
/// state (0 or 1) and can result in:
/// - `NoChange`: qubit stays at current value
/// - `Flip`: qubit flips to opposite value (X gate applied)
/// - `Leak`: qubit transitions to leaked state
///
/// This matches `GeneralNoiseModel`'s `p_meas_crosstalk_model` which uses
/// state-dependent transition probabilities.
#[derive(Debug, Clone)]
pub struct CrosstalkAction {
    transitions: crate::noise::CrosstalkTransitions,
}

impl CrosstalkAction {
    /// Create with the given transition model.
    #[must_use]
    pub fn new(transitions: crate::noise::CrosstalkTransitions) -> Self {
        Self { transitions }
    }

    /// Create with flip-only transitions (50% stay, 50% flip, no leakage).
    #[must_use]
    pub fn flip_only() -> Self {
        Self {
            transitions: crate::noise::CrosstalkTransitions::flip_only(),
        }
    }

    /// Create with symmetric transitions including leakage (1/3 each).
    #[must_use]
    pub fn symmetric_with_leakage() -> Self {
        Self {
            transitions: crate::noise::CrosstalkTransitions::symmetric_with_leakage(),
        }
    }
}

impl Default for CrosstalkAction {
    fn default() -> Self {
        Self::flip_only()
    }
}

impl GateAction for CrosstalkAction {
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        // Get the qubit's state from the measurement outcome if available,
        // otherwise sample randomly (unknown state).
        let state = ctx
            .current_outcome()
            .unwrap_or_else(|| rng.random::<bool>());

        let r: f64 = rng.random();
        let result = self.transitions.sample(state, r);

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

    fn name(&self) -> &'static str {
        "crosstalk"
    }
}

// ============================================================================
// Outcome Actions (for measurement noise)
// ============================================================================

/// Flip the measurement outcome (0 <-> 1).
///
/// This action is used in measurement noise models to apply
/// readout errors that flip the measurement result.
#[derive(Debug, Clone, Copy, Default)]
pub struct FlipOutcomeAction;

impl GateAction for FlipOutcomeAction {
    fn apply(
        &self,
        _qubit: QubitId,
        _ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        CompositeResponse::FlipOutcome
    }

    fn name(&self) -> &'static str {
        "flip_outcome"
    }
}

/// Force the measurement outcome to a specific value.
///
/// This is typically used for leaked qubits which should
/// always return a specific outcome (often 1).
#[derive(Debug, Clone, Copy)]
pub struct ForceOutcomeAction {
    value: bool,
}

impl ForceOutcomeAction {
    /// Create an action that forces the outcome to the given value.
    #[must_use]
    pub fn new(value: bool) -> Self {
        Self { value }
    }

    /// Force outcome to 0.
    #[must_use]
    pub fn zero() -> Self {
        Self::new(false)
    }

    /// Force outcome to 1.
    #[must_use]
    pub fn one() -> Self {
        Self::new(true)
    }
}

impl GateAction for ForceOutcomeAction {
    fn apply(
        &self,
        _qubit: QubitId,
        _ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        CompositeResponse::ForceOutcome(self.value)
    }

    fn name(&self) -> &'static str {
        "force_outcome"
    }
}

/// Randomize the measurement outcome (for leaked qubits).
///
/// This models the `MeasureLeaked` behavior where a leaked qubit
/// returns a random outcome since it's outside the computational basis.
/// With probability `prob_one`, the outcome is forced to 1.
#[derive(Debug, Clone, Copy)]
pub struct RandomOutcome {
    prob_one: f64,
}

impl RandomOutcome {
    /// Create an action that randomizes the outcome.
    #[must_use]
    pub fn new(prob_one: f64) -> Self {
        Self { prob_one }
    }

    /// Random with 50/50 probability (unbiased).
    #[must_use]
    pub fn uniform() -> Self {
        Self::new(0.5)
    }

    /// Biased towards 1 (models leaked qubits returning 1 more often).
    #[must_use]
    pub fn biased_one(bias: f64) -> Self {
        Self::new(bias)
    }
}

impl Default for RandomOutcome {
    fn default() -> Self {
        Self::uniform()
    }
}

impl GateAction for RandomOutcome {
    fn apply(
        &self,
        _qubit: QubitId,
        _ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        let r: f64 = rng.random();
        CompositeResponse::ForceOutcome(r < self.prob_one)
    }

    fn name(&self) -> &'static str {
        "random_outcome"
    }
}

/// Mark measurement as coming from a leaked qubit.
///
/// This action produces a `LeakedMeasurement` response which indicates
/// the outcome should be reported as 2 (the special leaked indicator)
/// rather than the actual measurement result.
///
/// This matches `MeasureLeaked` behavior in `GeneralNoiseModel`.
#[derive(Debug, Clone, Copy, Default)]
pub struct LeakedMeasurementAction;

impl GateAction for LeakedMeasurementAction {
    fn apply(
        &self,
        _qubit: QubitId,
        _ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        CompositeResponse::LeakedMeasurement
    }

    fn name(&self) -> &'static str {
        "leaked_measurement"
    }
}

/// Convenience functions for creating actions.
pub mod actions {
    use super::{
        AmplitudeDamping, BiasedAmplitudeDamping, CoherentRotation, CompositeResponse,
        CrosstalkAction, Emission, Erasure, ErasureWithReplacement, FlipOutcomeAction,
        ForceOutcomeAction, GateAction, GateType, IndependentEmissionWithPartnerDepolarize, Inject,
        InjectCoherentRZ, Leak, LeakedMeasurementAction, NoiseContext, Nothing, OverRotation,
        PartnerDepolarize, Pauli, PauliWeights, PecosRng, PrepFlip, PrepPhase, QubitId,
        RandomOutcome, SampleEmission, SampleEmissionWithProb, Seep, SkipGate, TwoQubitEmission,
        TwoQubitEmissionWithPartnerDepolarize, TwoQubitPauli, Unleak, ZZDephasing, ZZDephasingRate,
    };

    /// No-op action.
    #[must_use]
    pub fn nothing() -> Nothing {
        Nothing
    }

    /// Skip the current gate.
    #[must_use]
    pub fn skip_gate() -> SkipGate {
        SkipGate
    }

    /// Mark qubit as leaked.
    #[must_use]
    pub fn leak() -> Leak {
        Leak
    }

    /// Mark qubit as unleaked.
    #[must_use]
    pub fn unleak() -> Unleak {
        Unleak
    }

    /// Inject a specific gate.
    #[must_use]
    pub fn inject(gate_type: GateType) -> Inject {
        Inject::new(gate_type)
    }

    /// Inject an X gate.
    #[must_use]
    pub fn inject_x() -> Inject {
        Inject::x()
    }

    /// Inject a Y gate.
    #[must_use]
    pub fn inject_y() -> Inject {
        Inject::y()
    }

    /// Inject a Z gate.
    #[must_use]
    pub fn inject_z() -> Inject {
        Inject::z()
    }

    /// Random Pauli with uniform weights.
    #[must_use]
    pub fn pauli() -> Pauli {
        Pauli::uniform()
    }

    /// Random Pauli with custom weights.
    #[must_use]
    pub fn pauli_weighted(x: f64, y: f64, z: f64) -> Pauli {
        Pauli::new(PauliWeights::custom(x, y, z))
    }

    /// Dephasing (Z only).
    #[must_use]
    pub fn dephase() -> Pauli {
        Pauli::dephasing()
    }

    /// Seepage (unleak + random Pauli).
    #[must_use]
    pub fn seep() -> Seep {
        Seep::new()
    }

    // ========================================================================
    // Outcome Actions
    // ========================================================================

    /// Flip the measurement outcome (0 <-> 1).
    #[must_use]
    pub fn flip_outcome() -> FlipOutcomeAction {
        FlipOutcomeAction
    }

    /// Force the measurement outcome to a specific value.
    #[must_use]
    pub fn force_outcome(value: bool) -> ForceOutcomeAction {
        ForceOutcomeAction::new(value)
    }

    /// Force the measurement outcome to 0.
    #[must_use]
    pub fn force_zero() -> ForceOutcomeAction {
        ForceOutcomeAction::zero()
    }

    /// Force the measurement outcome to 1.
    #[must_use]
    pub fn force_one() -> ForceOutcomeAction {
        ForceOutcomeAction::one()
    }

    /// Randomize the measurement outcome (for leaked qubits).
    ///
    /// Returns a random outcome with 50/50 probability.
    #[must_use]
    pub fn random_outcome() -> RandomOutcome {
        RandomOutcome::uniform()
    }

    /// Randomize the measurement outcome with bias towards 1.
    ///
    /// This models leaked qubits which may return 1 more often.
    #[must_use]
    pub fn random_outcome_biased(prob_one: f64) -> RandomOutcome {
        RandomOutcome::new(prob_one)
    }

    /// Mark measurement as coming from a leaked qubit (outcome = 2).
    ///
    /// This matches `MeasureLeaked` behavior where leaked qubits return
    /// a special indicator (2) rather than a binary outcome.
    #[must_use]
    pub fn leaked_measurement() -> LeakedMeasurementAction {
        LeakedMeasurementAction
    }

    // ========================================================================
    // Crosstalk Actions
    // ========================================================================

    /// State-dependent crosstalk action (flip-only, no leakage).
    ///
    /// Uses 50% stay, 50% flip probabilities for both 0 and 1 states.
    #[must_use]
    pub fn crosstalk() -> CrosstalkAction {
        CrosstalkAction::flip_only()
    }

    /// State-dependent crosstalk action with leakage.
    ///
    /// Uses 1/3 stay, 1/3 flip, 1/3 leak for both states.
    #[must_use]
    pub fn crosstalk_with_leakage() -> CrosstalkAction {
        CrosstalkAction::symmetric_with_leakage()
    }

    /// State-dependent crosstalk action with custom transitions.
    #[must_use]
    pub fn crosstalk_transitions(
        transitions: crate::noise::CrosstalkTransitions,
    ) -> CrosstalkAction {
        CrosstalkAction::new(transitions)
    }

    // ========================================================================
    // Two-Qubit Actions
    // ========================================================================

    /// Correlated two-qubit Pauli with uniform weights.
    ///
    /// Samples from all 15 non-identity two-qubit Paulis.
    #[must_use]
    pub fn two_qubit_pauli() -> TwoQubitPauli {
        TwoQubitPauli::uniform()
    }

    /// Correlated two-qubit Pauli with custom weights.
    #[must_use]
    pub fn two_qubit_pauli_weighted(weights: crate::noise::TwoQubitPauliWeights) -> TwoQubitPauli {
        TwoQubitPauli::with_weights(weights)
    }

    /// Correlated two-qubit Pauli biased towards ZZ errors.
    #[must_use]
    pub fn two_qubit_pauli_zz_biased(zz_weight: f64) -> TwoQubitPauli {
        TwoQubitPauli::zz_biased(zz_weight)
    }

    /// Correlated two-qubit emission with uniform Pauli weights (no leakage).
    #[must_use]
    pub fn two_qubit_emission() -> TwoQubitEmission {
        TwoQubitEmission::uniform_pauli()
    }

    /// Correlated two-qubit emission with leakage.
    #[must_use]
    pub fn two_qubit_emission_with_leakage() -> TwoQubitEmission {
        TwoQubitEmission::uniform_with_leakage()
    }

    /// Correlated two-qubit emission with custom weights.
    #[must_use]
    pub fn two_qubit_emission_weighted(
        weights: crate::noise::TwoQubitEmissionWeights,
    ) -> TwoQubitEmission {
        TwoQubitEmission::with_weights(weights)
    }

    // ========================================================================
    // Single-Qubit Emission Actions
    // ========================================================================

    /// Emission error with uniform Pauli weights (no leakage).
    #[must_use]
    pub fn emission() -> Emission {
        Emission::pauli_only()
    }

    /// Emission error with specified leakage probability.
    ///
    /// Remaining probability is split equally among X, Y, Z.
    #[must_use]
    pub fn emission_with_leakage(leak_prob: f64) -> Emission {
        Emission::with_leakage(leak_prob)
    }

    /// Emission error with custom weights.
    #[must_use]
    pub fn emission_weighted(weights: crate::noise::SingleQubitEmissionWeights) -> Emission {
        Emission::with_weights(weights)
    }

    // ========================================================================
    // Two-Stage Sampling Actions
    // ========================================================================

    /// Sample emission and store the fired flag (for two-stage composite).
    ///
    /// Use this inside `prob(p, sample_emission())` for stage 1.
    /// The fired result can be queried in stage 2 using conditions
    /// like `i_fired()`, `partner_fired()`, or `partner_only_fired()`.
    #[must_use]
    pub fn sample_emission() -> SampleEmission {
        SampleEmission::new()
    }

    /// Sample emission with built-in probability (for two-stage composite).
    ///
    /// Unlike `sample_emission()` which should be wrapped in `prob()`,
    /// this action handles the probability sampling itself.
    #[must_use]
    pub fn sample_emission_with_prob(prob: f64) -> SampleEmissionWithProb {
        SampleEmissionWithProb::new(prob)
    }

    // ========================================================================
    // Coherent Dephasing
    // ========================================================================

    /// Inject coherent RZ dephasing with given rate.
    ///
    /// The angle is computed as `rate * idle_duration`.
    #[must_use]
    pub fn coherent_rz(rate: f64) -> InjectCoherentRZ {
        InjectCoherentRZ::new(rate)
    }

    // ========================================================================
    // Partner Depolarize (for two-qubit gates with leakage)
    // ========================================================================

    /// Apply depolarizing noise to the partner qubit in a two-qubit gate.
    ///
    /// This is used when one qubit is leaked: the leaked qubit gets no error,
    /// but its partner receives a random Pauli (X, Y, Z with equal probability).
    ///
    /// # Example
    ///
    /// ```
    /// # use pecos_neo::noise::composite::prelude::*;
    /// // In a two-qubit channel: if this qubit is leaked, depolarize partner
    /// let noise = when_leaked(partner_depolarize(), nothing());
    /// ```
    #[must_use]
    pub fn partner_depolarize() -> PartnerDepolarize {
        PartnerDepolarize::uniform()
    }

    /// Apply depolarizing noise to the partner qubit with custom weights.
    #[must_use]
    pub fn partner_depolarize_weighted(x: f64, y: f64, z: f64) -> PartnerDepolarize {
        PartnerDepolarize::with_weights(PauliWeights::custom(x, y, z))
    }

    // ========================================================================
    // Two-Qubit Emission with Partner Depolarize
    // ========================================================================

    /// Two-qubit emission with partner depolarizing.
    ///
    /// When used inside `prob(p, ...)`, this handles the full emission logic:
    /// - Both qubits that trigger this action are marked as leaked
    /// - If one emitted but the other didn't, the non-emitter gets depolarized
    ///
    /// # Example
    ///
    /// ```
    /// # use pecos_neo::noise::composite::prelude::*;
    /// // 1% chance of emission, with automatic partner depolarizing
    /// let noise = prob(0.01, two_qubit_emission_with_partner_depolarize());
    /// ```
    #[must_use]
    pub fn two_qubit_emission_with_partner_depolarize() -> TwoQubitEmissionWithPartnerDepolarize {
        TwoQubitEmissionWithPartnerDepolarize::new()
    }

    /// Independent emission sampling with partner depolarizing.
    ///
    /// This action samples emission independently for each qubit (unlike
    /// `two_qubit_emission_with_partner_depolarize` which assumes emission
    /// already occurred). Use this for full control over the emission process.
    ///
    /// # Example
    ///
    /// ```
    /// # use pecos_neo::noise::composite::prelude::*;
    /// // Each qubit has 1% independent emission probability
    /// let noise = independent_emission(0.01);
    /// ```
    #[must_use]
    pub fn independent_emission(emission_prob: f64) -> IndependentEmissionWithPartnerDepolarize {
        IndependentEmissionWithPartnerDepolarize::new(emission_prob)
    }

    // ========================================================================
    // Amplitude Damping (T1)
    // ========================================================================

    /// Amplitude damping (T1 relaxation) with 50/50 X/Z approximation.
    #[must_use]
    pub fn amplitude_damping() -> AmplitudeDamping {
        AmplitudeDamping::new()
    }

    /// Biased amplitude damping with custom X probability.
    #[must_use]
    pub fn amplitude_damping_biased(x_probability: f64) -> BiasedAmplitudeDamping {
        BiasedAmplitudeDamping::new(x_probability)
    }

    // ========================================================================
    // Coherent Errors
    // ========================================================================

    /// Coherent RX rotation error.
    #[must_use]
    pub fn coherent_rx(angle: f64) -> CoherentRotation {
        CoherentRotation::rx(angle)
    }

    /// Coherent RY rotation error.
    #[must_use]
    pub fn coherent_ry(angle: f64) -> CoherentRotation {
        CoherentRotation::ry(angle)
    }

    /// Coherent RZ rotation error (phase error).
    #[must_use]
    pub fn coherent_rz_fixed(angle: f64) -> CoherentRotation {
        CoherentRotation::rz(angle)
    }

    /// Over-rotation error (fraction of gate angle).
    ///
    /// The error angle is `gate_angle * fraction`.
    #[must_use]
    pub fn over_rotation(gate_type: GateType, fraction: f64) -> OverRotation {
        OverRotation::new(gate_type, fraction)
    }

    /// RZ over-rotation error.
    #[must_use]
    pub fn over_rotation_rz(fraction: f64) -> OverRotation {
        OverRotation::rz(fraction)
    }

    // ========================================================================
    // Correlated Dephasing (ZZ)
    // ========================================================================

    /// ZZ dephasing with fixed angle.
    #[must_use]
    pub fn zz_dephasing(angle: f64) -> ZZDephasing {
        ZZDephasing::new(angle)
    }

    /// ZZ dephasing with rate (angle = rate * duration).
    #[must_use]
    pub fn zz_dephasing_rate(rate: f64) -> ZZDephasingRate {
        ZZDephasing::from_rate(rate)
    }

    // ========================================================================
    // Preparation Errors
    // ========================================================================

    /// Preparation bit-flip error (applies X).
    #[must_use]
    pub fn prep_flip() -> PrepFlip {
        PrepFlip::new()
    }

    /// Preparation phase error (applies Z).
    #[must_use]
    pub fn prep_phase() -> PrepPhase {
        PrepPhase::new()
    }

    // ========================================================================
    // Erasure Errors
    // ========================================================================

    /// Erasure error (heralded qubit loss).
    #[must_use]
    pub fn erasure() -> Erasure {
        Erasure::new()
    }

    /// Erasure with replacement (reset to random state).
    #[must_use]
    pub fn erasure_with_replacement() -> ErasureWithReplacement {
        ErasureWithReplacement::new()
    }

    // ========================================================================
    // Channel Adapter
    // ========================================================================

    /// Adapter that wraps a `NoiseChannel` to be used as a composite primitive.
    ///
    /// This enables using traditional channel implementations as building blocks
    /// within composite noise decision trees. The adapter reconstructs the noise event
    /// from the current context and delegates to the wrapped channel.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::prelude::*;
    /// use pecos_neo::noise::SingleQubitChannel;
    ///
    /// // Use a traditional channel inside a composite decision tree
    /// let noise = seq![
    ///     skip_if_leaked(),
    ///     prob(0.5, channel_action(SingleQubitChannel::depolarizing(1.0))),
    /// ];
    /// ```
    ///
    /// # Limitations
    ///
    /// - The adapter creates a synthetic event for the current qubit only
    /// - For two-qubit channels, use with caution as the event may not have
    ///   full partner qubit information depending on context
    /// - The channel's `responds_to` filter is bypassed (assumed to match)
    pub struct ChannelAdapter {
        channel: Box<dyn crate::noise::NoiseChannel>,
        name: String,
    }

    impl Clone for ChannelAdapter {
        fn clone(&self) -> Self {
            Self {
                channel: self.channel.clone_box(),
                name: self.name.clone(),
            }
        }
    }

    impl ChannelAdapter {
        /// Create a new channel adapter.
        pub fn new<C: crate::noise::NoiseChannel + 'static>(channel: C) -> Self {
            let name = channel.name().to_string();
            Self {
                channel: Box::new(channel),
                name,
            }
        }

        /// Get the name of the wrapped channel.
        #[must_use]
        pub fn channel_name(&self) -> &str {
            &self.name
        }
    }

    impl std::fmt::Debug for ChannelAdapter {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ChannelAdapter")
                .field("name", &self.name)
                .finish_non_exhaustive()
        }
    }

    impl GateAction for ChannelAdapter {
        fn apply(
            &self,
            qubit: QubitId,
            ctx: &mut NoiseContext,
            rng: &mut PecosRng,
        ) -> CompositeResponse {
            use crate::noise::NoiseEvent;

            // Get current gate info from context
            let gate_info = ctx.current_gate();
            let gate_qubits = ctx.current_gate_qubits();

            // Build the event based on available context
            let qubits: Vec<QubitId> = if gate_qubits.is_empty() {
                vec![qubit]
            } else {
                gate_qubits.to_vec()
            };

            let angles: Vec<pecos_core::Angle64> =
                gate_info.map(|g| g.angles.to_vec()).unwrap_or_default();

            let gate_type = gate_info.map_or(GateType::I, |g| g.gate_type);

            // Create the event using helper constructor
            let event = NoiseEvent::after_gate(gate_type, &qubits, &angles);

            // Apply the wrapped channel
            let response = self.channel.apply(&event, ctx, rng);

            // Convert NoiseResponse to CompositeResponse
            convert_noise_response(response, qubit)
        }

        fn name(&self) -> &'static str {
            "channel_adapter"
        }
    }

    /// Convert a `NoiseResponse` to a `CompositeResponse`.
    ///
    /// This handles the mapping between the two response types.
    fn convert_noise_response(
        response: crate::noise::NoiseResponse,
        qubit: QubitId,
    ) -> CompositeResponse {
        use crate::noise::NoiseResponse;

        match response {
            NoiseResponse::None => CompositeResponse::None,
            NoiseResponse::SkipGate => CompositeResponse::SkipGate,
            NoiseResponse::InjectGates(gates) => CompositeResponse::InjectGates(gates.to_vec()),
            NoiseResponse::MarkLeaked(qs) => {
                if qs.contains(&qubit) {
                    CompositeResponse::Leak
                } else {
                    // Can't mark other qubits as leaked from here
                    CompositeResponse::None
                }
            }
            NoiseResponse::MarkUnleaked(qs) => {
                if qs.contains(&qubit) {
                    CompositeResponse::Unleak
                } else {
                    CompositeResponse::None
                }
            }
            NoiseResponse::FlipOutcomes(qs) => {
                if qs.contains(&qubit) {
                    CompositeResponse::FlipOutcome
                } else {
                    CompositeResponse::None
                }
            }
            NoiseResponse::LeakedMeasurement(qs) => {
                if qs.contains(&qubit) {
                    CompositeResponse::LeakedMeasurement
                } else {
                    CompositeResponse::None
                }
            }
            NoiseResponse::ForceOutcomes(forced) => {
                // Find if this qubit has a forced outcome
                forced
                    .iter()
                    .find(|(q, _)| *q == qubit)
                    .map_or(CompositeResponse::None, |(_, value)| {
                        CompositeResponse::ForceOutcome(*value)
                    })
            }
            NoiseResponse::Multiple(responses) => {
                let converted: Vec<_> = responses
                    .into_iter()
                    .map(|r| convert_noise_response(r, qubit))
                    .filter(|r| !matches!(r, CompositeResponse::None))
                    .collect();

                match converted.len() {
                    0 => CompositeResponse::None,
                    1 => converted
                        .into_iter()
                        .next()
                        .expect("len is 1, so next() returns Some"),
                    _ => CompositeResponse::Multiple(converted),
                }
            }
        }
    }

    /// Wrap a traditional `NoiseChannel` as a composite primitive action.
    ///
    /// This allows using traditional channel implementations as building blocks
    /// within composite noise decision trees.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::noise::composite::prelude::*;
    /// use pecos_neo::noise::SingleQubitChannel;
    ///
    /// // Use a traditional channel inside a composite decision tree
    /// let noise = seq![
    ///     skip_if_leaked(),
    ///     prob(0.5, channel_action(SingleQubitChannel::depolarizing(1.0))),
    /// ];
    /// ```
    #[must_use]
    pub fn channel_action<C: crate::noise::NoiseChannel + 'static>(channel: C) -> ChannelAdapter {
        ChannelAdapter::new(channel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_context() -> NoiseContext {
        NoiseContext::new()
    }

    #[test]
    fn test_nothing_action() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = Nothing.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(response.is_none());
    }

    #[test]
    fn test_skip_gate_action() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = SkipGate.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(response.skips_gate());
    }

    #[test]
    fn test_leak_action() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        assert!(!ctx.is_leaked(QubitId(0)));

        let response = Leak.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(response.causes_leak());
        assert!(ctx.is_leaked(QubitId(0)));
    }

    #[test]
    fn test_inject_action() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = Inject::x().apply(QubitId(0), &mut ctx, &mut rng);
        let gates = response.collect_gates();

        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::X);
        assert_eq!(gates[0].qubits[0], QubitId(0));
    }

    #[test]
    fn test_pauli_action_distribution() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);
        let pauli = Pauli::uniform();

        let mut x_count = 0;
        let mut y_count = 0;
        let mut z_count = 0;

        for _ in 0..3000 {
            let response = pauli.apply(QubitId(0), &mut ctx, &mut rng);
            let gates = response.collect_gates();
            assert_eq!(gates.len(), 1);

            match gates[0].gate_type {
                GateType::X => x_count += 1,
                GateType::Y => y_count += 1,
                GateType::Z => z_count += 1,
                _ => panic!("Unexpected gate type"),
            }
        }

        // Should be roughly uniform (1/3 each)
        let total = 3000.0;
        assert!((f64::from(x_count) / total - 1.0 / 3.0).abs() < 0.05);
        assert!((f64::from(y_count) / total - 1.0 / 3.0).abs() < 0.05);
        assert!((f64::from(z_count) / total - 1.0 / 3.0).abs() < 0.05);
    }

    #[test]
    fn test_seep_action() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // Mark as leaked first
        ctx.mark_leaked(QubitId(0));
        assert!(ctx.is_leaked(QubitId(0)));

        // Seep should unleak
        let _response = Seep::new().apply(QubitId(0), &mut ctx, &mut rng);
        assert!(!ctx.is_leaked(QubitId(0)));
    }

    #[test]
    fn test_two_qubit_pauli_uniform() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);
        let action = TwoQubitPauli::uniform();

        // Run many times and verify we get a mix of Paulis
        let mut gate_counts = std::collections::HashMap::new();

        for _ in 0..1500 {
            let response = action.apply(QubitId(0), &mut ctx, &mut rng);
            let gates = response.collect_gates();

            // Should get 0-1 gates (since we only apply to first qubit without context)
            for gate in gates {
                *gate_counts.entry(gate.gate_type).or_insert(0) += 1;
            }
            ctx.clear_correlation();
        }

        // Should have X, Y, Z in roughly uniform distribution
        // (the first qubit portion of two-qubit Paulis)
        assert!(gate_counts.contains_key(&GateType::X));
        assert!(gate_counts.contains_key(&GateType::Y));
        assert!(gate_counts.contains_key(&GateType::Z));
    }

    #[test]
    fn test_two_qubit_pauli_correlated() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);
        let action = TwoQubitPauli::uniform();

        // Simulate how CompositeChannel processes two-qubit gates
        let qubits = [QubitId(0), QubitId(1)];

        // Track pairs of Paulis applied
        let mut pairs: Vec<(Option<GateType>, Option<GateType>)> = Vec::new();

        for _ in 0..1000 {
            // Process first qubit
            ctx.set_current_qubit_index(0, &qubits);
            let response1 = action.apply(QubitId(0), &mut ctx, &mut rng);
            let gates1 = response1.collect_gates();
            let pauli1 = gates1.first().map(|g| g.gate_type);

            // Process second qubit (should use stored correlation)
            ctx.set_current_qubit_index(1, &qubits);
            let response2 = action.apply(QubitId(1), &mut ctx, &mut rng);
            let gates2 = response2.collect_gates();
            let pauli2 = gates2.first().map(|g| g.gate_type);

            pairs.push((pauli1, pauli2));
            ctx.clear_correlation();
        }

        // Verify we get correlated pairs (not just independent sampling)
        // Count how many times we get XX, YY, ZZ (correlated) vs mixed
        let mut same_count = 0;
        let mut mixed_count = 0;
        for (p1, p2) in &pairs {
            if p1.is_some() && p2.is_some() && p1 == p2 {
                same_count += 1;
            } else if p1.is_some() || p2.is_some() {
                mixed_count += 1;
            }
        }

        // With uniform 15-Pauli weights:
        // XX, YY, ZZ each have weight 1/15
        // So ~3/15 = 20% should be same-Pauli pairs
        // This tests that the correlation is working
        let total = same_count + mixed_count;
        if total > 0 {
            let same_rate = f64::from(same_count) / f64::from(total);
            // Allow wide tolerance for statistical variation
            assert!(
                same_rate > 0.1 && same_rate < 0.4,
                "Expected ~20% same-Pauli pairs, got {:.1}%",
                same_rate * 100.0
            );
        }
    }

    #[test]
    fn test_emission_pauli_only() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);
        let action = Emission::pauli_only();

        let mut leak_count = 0;
        let mut pauli_count = 0;

        for _ in 0..1000 {
            let response = action.apply(QubitId(0), &mut ctx, &mut rng);
            if response.causes_leak() {
                leak_count += 1;
            } else if !response.collect_gates().is_empty() {
                pauli_count += 1;
            }
            ctx.mark_unleaked(QubitId(0)); // Reset for next iteration
        }

        // Should have no leakage and all Pauli
        assert_eq!(leak_count, 0);
        assert_eq!(pauli_count, 1000);
    }

    #[test]
    fn test_emission_with_leakage() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);
        let action = Emission::with_leakage(0.25); // 25% leakage

        let mut leak_count = 0;

        for _ in 0..1000 {
            let response = action.apply(QubitId(0), &mut ctx, &mut rng);
            if response.causes_leak() {
                leak_count += 1;
            }
            ctx.mark_unleaked(QubitId(0)); // Reset for next iteration
        }

        // Should be roughly 25% leakage, 75% Pauli
        let leak_rate = f64::from(leak_count) / 1000.0;
        assert!(
            (leak_rate - 0.25).abs() < 0.05,
            "Expected ~25% leakage, got {leak_rate}"
        );
    }

    #[test]
    fn test_coherent_rz_with_duration() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);
        let action = InjectCoherentRZ::new(0.1); // 0.1 rad per time unit

        // Set idle duration
        ctx.set_current_idle(pecos_core::TimeUnits::new(5)); // 5 time units

        let response = action.apply(QubitId(0), &mut ctx, &mut rng);
        let gates = response.collect_gates();

        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].gate_type, GateType::RZ);
        // Angle should be 0.1 * 5 = 0.5 rad
        let angle = gates[0].angles[0].to_radians();
        assert!(
            (angle - 0.5).abs() < 1e-10,
            "Expected angle 0.5, got {angle}"
        );
    }

    #[test]
    fn test_coherent_rz_no_duration() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);
        let action = InjectCoherentRZ::new(0.1);

        // No idle duration set
        let response = action.apply(QubitId(0), &mut ctx, &mut rng);

        // Should return None (no gate injected for zero duration)
        assert!(response.is_none());
    }

    // ========================================================================
    // Crosstalk Action Tests
    // ========================================================================

    #[test]
    fn test_crosstalk_action_flip_only() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);
        let action = CrosstalkAction::flip_only();

        // Run many times and count outcomes
        let mut no_change = 0;
        let mut flip = 0;
        let mut leak = 0;

        for _ in 0..1000 {
            let response = action.apply(QubitId(0), &mut ctx, &mut rng);
            if response.is_none() {
                no_change += 1;
            } else if response.causes_leak() {
                leak += 1;
            } else if !response.collect_gates().is_empty() {
                flip += 1;
            }
            ctx.mark_unleaked(QubitId(0)); // Reset for next iteration
        }

        // With flip_only: 50% stay, 50% flip, 0% leak
        let stay_rate = f64::from(no_change) / 1000.0;
        let flip_rate = f64::from(flip) / 1000.0;

        assert!(
            (stay_rate - 0.5).abs() < 0.1,
            "Expected ~50% no change, got {stay_rate}"
        );
        assert!(
            (flip_rate - 0.5).abs() < 0.1,
            "Expected ~50% flip, got {flip_rate}"
        );
        assert_eq!(leak, 0, "Expected 0 leakage");
    }

    #[test]
    fn test_crosstalk_action_with_leakage() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);
        let action = CrosstalkAction::symmetric_with_leakage();

        // Run many times and count outcomes
        let mut no_change = 0;
        let mut flip = 0;
        let mut leak = 0;

        for _ in 0..3000 {
            let response = action.apply(QubitId(0), &mut ctx, &mut rng);
            if response.is_none() {
                no_change += 1;
            } else if response.causes_leak() {
                leak += 1;
            } else if !response.collect_gates().is_empty() {
                flip += 1;
            }
            ctx.mark_unleaked(QubitId(0)); // Reset for next iteration
        }

        // With symmetric_with_leakage: 1/3 each
        let total = 3000.0;
        let stay_rate = f64::from(no_change) / total;
        let flip_rate = f64::from(flip) / total;
        let leak_rate = f64::from(leak) / total;

        assert!(
            (stay_rate - 0.333).abs() < 0.05,
            "Expected ~33% no change, got {stay_rate}"
        );
        assert!(
            (flip_rate - 0.333).abs() < 0.05,
            "Expected ~33% flip, got {flip_rate}"
        );
        assert!(
            (leak_rate - 0.333).abs() < 0.05,
            "Expected ~33% leakage, got {leak_rate}"
        );
    }

    #[test]
    fn test_crosstalk_action_uses_outcome_state() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);

        // Create asymmetric transitions:
        // From 0: 100% flip
        // From 1: 100% stay
        let action = CrosstalkAction::new(crate::noise::CrosstalkTransitions::custom(
            0.0, 1.0, 0.0, 1.0, 0.0, 0.0,
        ));

        // Set outcome to 0 -> should always flip (X gate)
        ctx.set_current_outcome(false);
        for _ in 0..10 {
            let response = action.apply(QubitId(0), &mut ctx, &mut rng);
            assert!(!response.is_none(), "With outcome=0, should always flip");
        }

        // Set outcome to 1 -> should always stay (no change)
        ctx.set_current_outcome(true);
        for _ in 0..10 {
            let response = action.apply(QubitId(0), &mut ctx, &mut rng);
            assert!(response.is_none(), "With outcome=1, should always stay");
        }
    }

    // ========================================================================
    // Random Outcome Tests
    // ========================================================================

    #[test]
    fn test_random_outcome_uniform() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);
        let action = RandomOutcome::uniform();

        let mut ones = 0;

        for _ in 0..1000 {
            let response = action.apply(QubitId(0), &mut ctx, &mut rng);
            match response.forces_outcome() {
                Some(true) => ones += 1,
                Some(false) => {}
                None => panic!("Expected ForceOutcome response"),
            }
        }

        // Should be roughly 50/50
        let one_rate = f64::from(ones) / 1000.0;
        assert!(
            (one_rate - 0.5).abs() < 0.1,
            "Expected ~50% ones, got {one_rate}"
        );
    }

    #[test]
    fn test_random_outcome_biased() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);
        let action = RandomOutcome::biased_one(0.8); // 80% towards 1

        let mut ones = 0;

        for _ in 0..1000 {
            let response = action.apply(QubitId(0), &mut ctx, &mut rng);
            if response.forces_outcome() == Some(true) {
                ones += 1;
            }
        }

        // Should be roughly 80%
        let one_rate = f64::from(ones) / 1000.0;
        assert!(
            (one_rate - 0.8).abs() < 0.1,
            "Expected ~80% ones, got {one_rate}"
        );
    }

    // ========================================================================
    // Two-Stage Sample Emission Tests
    // ========================================================================

    #[test]
    fn test_sample_emission_sets_fired_flag() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);
        let qubits = [QubitId(0), QubitId(1)];

        // Set up as qubit 0
        ctx.set_current_qubit_index(0, &qubits);

        // Sample emission should set fired flag and mark as leaked
        let response = SampleEmission::new().apply(QubitId(0), &mut ctx, &mut rng);

        assert!(ctx.is_fired(0), "Should have set fired flag for qubit 0");
        assert!(
            ctx.is_leaked(QubitId(0)),
            "Should have marked qubit as leaked"
        );
        assert!(response.causes_leak(), "Should return Leak response");
    }

    #[test]
    fn test_sample_emission_with_prob() {
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);
        let action = SampleEmissionWithProb::new(0.5); // 50% emission probability

        let mut fired_count = 0;

        for i in 0..1000 {
            ctx.set_current_qubit_index(0, &[QubitId(i)]);
            ctx.clear_fired_flags();

            action.apply(QubitId(i), &mut ctx, &mut rng);

            if ctx.is_fired(0) {
                fired_count += 1;
            }
        }

        // Should be roughly 50%
        let fired_rate = f64::from(fired_count) / 1000.0;
        assert!(
            (fired_rate - 0.5).abs() < 0.1,
            "Expected ~50% fired, got {fired_rate}"
        );
    }

    #[test]
    fn test_two_stage_emission_flow() {
        // Simulate a complete two-stage composite for a two-qubit gate
        let mut ctx = make_test_context();
        let mut rng = PecosRng::seed_from_u64(42);
        let qubits = [QubitId(0), QubitId(1)];

        // === Stage 1: Sample emission for each qubit ===

        // Qubit 0: sample with 100% probability (always fires for testing)
        let action0 = SampleEmissionWithProb::new(1.0);
        ctx.set_current_qubit_index(0, &qubits);
        action0.apply(QubitId(0), &mut ctx, &mut rng);

        assert!(ctx.is_fired(0), "Qubit 0 should have fired");

        // Qubit 1: sample with 0% probability (never fires for testing)
        let action1 = SampleEmissionWithProb::new(0.0);
        ctx.set_current_qubit_index(1, &qubits);
        action1.apply(QubitId(1), &mut ctx, &mut rng);

        assert!(!ctx.is_fired(1), "Qubit 1 should NOT have fired");

        // === Stage 2: Check cross-conditions ===

        // From qubit 0's perspective: I fired, partner didn't
        ctx.set_current_qubit_index(0, &qubits);
        assert!(ctx.current_qubit_fired(), "Qubit 0 should show as fired");
        assert!(
            !ctx.partner_fired(),
            "Qubit 0's partner should NOT show as fired"
        );

        // From qubit 1's perspective: I didn't fire, partner fired
        ctx.set_current_qubit_index(1, &qubits);
        assert!(
            !ctx.current_qubit_fired(),
            "Qubit 1 should NOT show as fired"
        );
        assert!(
            ctx.partner_fired(),
            "Qubit 1's partner SHOULD show as fired"
        );
    }
}
