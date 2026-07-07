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

//! Two-qubit gate noise channel.
//!
//! This is a traditional standalone channel implementation. For composable,
//! declarative noise models with conditional logic, see `CompositeChannel` in
//! `pecos_neo::noise::composite::prelude`.
//!
//! ## When to use this vs `CompositeChannel`
//!
//! **Use `TwoQubitChannel` when:**
//! - You want a simple, direct noise model
//! - Performance is critical (no primitive tree traversal)
//! - The built-in options (depolarizing, angle scaling, emission) suffice
//!
//! **Use `CompositeChannel` when:**
//! - You need complex conditional logic (partner leaked, cross-qubit conditions)
//! - You want two-stage processing for correlated effects
//! - You need custom branching or sampling behavior
//!
//! Handles depolarizing and Pauli noise on two-qubit gates, including
//! angle-dependent error scaling for parameterized gates like RZZ.

use super::{
    NoiseChannel, NoiseContext, NoiseEvent, NoiseResponse, TwoQubitEmissionWeights,
    TwoQubitPauliWeights,
};
use crate::command::{GateCommand, GateType};
use pecos_core::Angle64;
use pecos_random::PecosRng;
use rand::RngExt;
use smallvec::SmallVec;

/// Angle-dependent error scaling parameters.
///
/// Supports both symmetric and asymmetric scaling with the full polynomial form:
///
/// **Formula**: `p(θ) = a + b*|θ/π| + c*|θ/π|^d`
///
/// where:
/// - `a` = offset (constant term)
/// - `b` = linear coefficient
/// - `c` = scale (power term coefficient)
/// - `d` = power exponent
///
/// **Asymmetric**: Different coefficients for positive vs negative angles:
/// - Negative: `neg_offset + neg_linear*|θ/π| + neg_scale*|θ/π|^power`
/// - Positive: `pos_offset + pos_linear*|θ/π| + pos_scale*|θ/π|^power`
/// - Zero: average of offsets
///
/// This matches `GeneralNoiseModel`'s `p2_angle_a/b/c/d/power` parameters.
#[derive(Debug, Clone, Copy)]
pub struct AngleScaling {
    /// Constant term for negative angles (a).
    pub neg_offset: f64,
    /// Linear coefficient for negative angles (b).
    pub neg_linear: f64,
    /// Power term coefficient for negative angles (c).
    pub neg_scale: f64,
    /// Constant term for positive angles (a).
    pub pos_offset: f64,
    /// Linear coefficient for positive angles (b).
    pub pos_linear: f64,
    /// Power term coefficient for positive angles (c).
    pub pos_scale: f64,
    /// Power exponent for angle scaling (d).
    pub power: f64,
}

impl Default for AngleScaling {
    fn default() -> Self {
        Self::constant()
    }
}

impl AngleScaling {
    /// No angle dependence (constant error rate = 1.0).
    #[must_use]
    pub fn constant() -> Self {
        Self {
            neg_offset: 1.0,
            neg_linear: 0.0,
            neg_scale: 0.0,
            pos_offset: 1.0,
            pos_linear: 0.0,
            pos_scale: 0.0,
            power: 1.0,
        }
    }

    /// Symmetric linear scaling with angle.
    ///
    /// Error scales as |theta/pi|.
    #[must_use]
    pub fn linear() -> Self {
        Self {
            neg_offset: 0.0,
            neg_linear: 1.0,
            neg_scale: 0.0,
            pos_offset: 0.0,
            pos_linear: 1.0,
            pos_scale: 0.0,
            power: 1.0,
        }
    }

    /// Symmetric quadratic scaling with angle.
    ///
    /// Error scales as (theta/pi)^2.
    #[must_use]
    pub fn quadratic() -> Self {
        Self {
            neg_offset: 0.0,
            neg_linear: 0.0,
            neg_scale: 1.0,
            pos_offset: 0.0,
            pos_linear: 0.0,
            pos_scale: 1.0,
            power: 2.0,
        }
    }

    /// Symmetric polynomial scaling: `a + b*|θ/π| + c*|θ/π|^d`.
    ///
    /// This matches `GeneralNoiseModel`'s angle scaling formula.
    ///
    /// # Arguments
    /// * `a` - Constant term (offset)
    /// * `b` - Linear coefficient
    /// * `c` - Power term coefficient
    /// * `d` - Power exponent
    #[must_use]
    pub fn polynomial(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self {
            neg_offset: a,
            neg_linear: b,
            neg_scale: c,
            pos_offset: a,
            pos_linear: b,
            pos_scale: c,
            power: d,
        }
    }

    /// Create asymmetric scaling with different coefficients for +/- angles.
    ///
    /// Formula for each sign:
    /// `offset + linear*|θ/π| + scale*|θ/π|^power`
    #[must_use]
    pub fn asymmetric(
        neg_offset: f64,
        neg_linear: f64,
        neg_scale: f64,
        pos_offset: f64,
        pos_linear: f64,
        pos_scale: f64,
        power: f64,
    ) -> Self {
        Self {
            neg_offset,
            neg_linear,
            neg_scale,
            pos_offset,
            pos_linear,
            pos_scale,
            power,
        }
    }

    /// Create from `GeneralNoiseModel` parameters.
    ///
    /// Maps the `p2_angle_*` parameters to `AngleScaling`:
    /// - `a` → constant offset
    /// - `b` → linear coefficient
    /// - `c` → power term coefficient
    /// - `d` → power exponent
    ///
    /// This creates symmetric scaling (same for positive and negative angles).
    #[must_use]
    pub fn from_general_params(a: f64, b: f64, c: f64, d: f64) -> Self {
        Self::polynomial(a, b, c, d)
    }

    /// Set the power exponent.
    #[must_use]
    pub fn with_power(mut self, power: f64) -> Self {
        self.power = power;
        self
    }

    /// Set the constant offset (symmetric).
    #[must_use]
    pub fn with_offset(mut self, offset: f64) -> Self {
        self.neg_offset = offset;
        self.pos_offset = offset;
        self
    }

    /// Set the linear coefficient (symmetric).
    #[must_use]
    pub fn with_linear(mut self, linear: f64) -> Self {
        self.neg_linear = linear;
        self.pos_linear = linear;
        self
    }

    /// Set the power term coefficient (symmetric).
    #[must_use]
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.neg_scale = scale;
        self.pos_scale = scale;
        self
    }

    /// Calculate the scaling factor for a given angle.
    ///
    /// Uses signed radians in [-pi, pi] for asymmetric scaling.
    /// The magnitude is normalized by pi before applying power.
    ///
    /// Formula: `offset + linear*|θ/π| + scale*|θ/π|^power`
    #[must_use]
    pub fn scale(&self, angle: Angle64) -> f64 {
        // Use signed radians to distinguish positive/negative angles
        let theta = angle.to_radians_signed();
        // Normalize magnitude by pi to get value in [0, 1] range
        let theta_norm = theta.abs() / std::f64::consts::PI;
        let theta_linear = theta_norm;
        let theta_power = theta_norm.powf(self.power);

        if theta < 0.0 {
            self.neg_offset + self.neg_linear * theta_linear + self.neg_scale * theta_power
        } else if theta > 0.0 {
            self.pos_offset + self.pos_linear * theta_linear + self.pos_scale * theta_power
        } else {
            // Zero angle: average of offsets
            (self.neg_offset + self.pos_offset) * 0.5
        }
    }
}

/// Noise channel for two-qubit gates.
///
/// Models two types of errors:
/// 1. Pauli errors with configurable two-qubit Pauli distribution
/// 2. Emission errors that can cause Pauli errors AND/OR leakage
///
/// Applies two-qubit Pauli errors after two-qubit gates.
#[derive(Debug, Clone)]
pub struct TwoQubitChannel {
    /// Base probability of any error occurring.
    pub error_probability: f64,

    /// Angle scaling for parameterized gates.
    pub angle_scaling: AngleScaling,

    /// Distribution of two-qubit Pauli errors.
    pub pauli_weights: TwoQubitPauliWeights,

    /// Fraction of errors that are emission errors (vs Pauli errors).
    ///
    /// When an error occurs:
    /// - With probability `emission_ratio`, it's an emission error
    /// - Otherwise, it's a Pauli error from `pauli_weights`
    pub emission_ratio: f64,

    /// Distribution of emission errors (Pauli gates and/or leakage).
    ///
    /// When an emission error occurs, sample from this distribution
    /// to determine whether to apply Pauli errors and/or cause leakage.
    pub emission_weights: TwoQubitEmissionWeights,

    /// Seepage probability for leaked qubits.
    pub seepage_probability: f64,

    /// Idle noise rate applied after two-qubit gates.
    ///
    /// If non-zero, applies stochastic Z errors to involved qubits
    /// after the gate (for memory sweeping).
    pub idle_rate: f64,

    // Precomputed probability thresholds for fast sampling
    seepage_threshold: u64,
    emission_threshold: u64,
    idle_threshold: u64,
}

impl Default for TwoQubitChannel {
    fn default() -> Self {
        Self {
            error_probability: 0.0,
            angle_scaling: AngleScaling::default(),
            pauli_weights: TwoQubitPauliWeights::uniform(),
            emission_ratio: 0.0,
            emission_weights: TwoQubitEmissionWeights::uniform_pauli(),
            seepage_probability: 0.0,
            idle_rate: 0.0,
            seepage_threshold: 0,
            emission_threshold: 0,
            idle_threshold: 0,
        }
    }
}

impl TwoQubitChannel {
    /// Create a new channel with all parameters specified.
    ///
    /// Precomputes probability thresholds for faster sampling.
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        error_probability: f64,
        angle_scaling: AngleScaling,
        pauli_weights: TwoQubitPauliWeights,
        emission_ratio: f64,
        emission_weights: TwoQubitEmissionWeights,
        seepage_probability: f64,
        idle_rate: f64,
    ) -> Self {
        Self {
            error_probability,
            angle_scaling,
            pauli_weights,
            emission_ratio,
            emission_weights,
            seepage_probability,
            idle_rate,
            seepage_threshold: PecosRng::probability_threshold(seepage_probability),
            emission_threshold: PecosRng::probability_threshold(emission_ratio),
            idle_threshold: PecosRng::probability_threshold(idle_rate),
        }
    }

    /// Create a depolarizing noise channel.
    #[must_use]
    pub fn depolarizing(p: f64) -> Self {
        Self {
            error_probability: p,
            ..Default::default()
        }
    }

    /// Set angle-dependent scaling.
    #[must_use]
    pub fn with_angle_scaling(mut self, scaling: AngleScaling) -> Self {
        self.angle_scaling = scaling;
        self
    }

    /// Set the emission error ratio with uniform leakage on both qubits.
    ///
    /// This is a convenience method that configures emission weights
    /// to include leakage with the specified probability on both qubits.
    #[must_use]
    pub fn with_leakage(mut self, ratio: f64) -> Self {
        self.emission_ratio = ratio;
        self.emission_threshold = PecosRng::probability_threshold(ratio);
        // Use uniform with leakage distribution
        self.emission_weights = TwoQubitEmissionWeights::uniform_with_leakage();
        self
    }

    /// Set the emission error ratio with custom emission weights.
    #[must_use]
    pub fn with_emission_weights(mut self, ratio: f64, weights: TwoQubitEmissionWeights) -> Self {
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

    /// Set the Pauli error distribution.
    #[must_use]
    pub fn with_pauli_weights(mut self, weights: TwoQubitPauliWeights) -> Self {
        self.pauli_weights = weights;
        self
    }

    /// Set the idle noise rate applied after two-qubit gates.
    ///
    /// This models memory errors (T1/T2) that occur during the gate.
    #[must_use]
    pub fn with_idle_rate(mut self, rate: f64) -> Self {
        self.idle_rate = rate;
        self.idle_threshold = PecosRng::probability_threshold(rate);
        self
    }

    /// Scale the error probability by a factor.
    ///
    /// This multiplies the current error probability by `scale`.
    /// Useful for globally adjusting noise levels.
    #[must_use]
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.error_probability *= scale;
        self
    }

    /// Get the effective error probability for a gate, considering angle scaling.
    fn effective_probability(&self, gate_type: GateType, angles: &[Angle64]) -> f64 {
        if angles.is_empty() {
            return self.error_probability;
        }

        // Only scale for parameterized two-qubit gates
        let scale = match gate_type {
            GateType::RZZ | GateType::RXX | GateType::RYY | GateType::CRZ => {
                self.angle_scaling.scale(angles[0])
            }
            _ => 1.0,
        };

        (self.error_probability * scale).min(1.0)
    }
}

impl NoiseChannel for TwoQubitChannel {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        if self.error_probability <= 0.0 {
            return false;
        }
        // Only respond to unitary two-qubit gates.
        // Non-unitary two-qubit operations (if any) should not get
        // gate depolarizing noise.
        match event {
            NoiseEvent::BeforeGate { gate_type, .. } | NoiseEvent::AfterGate { gate_type, .. } => {
                gate_type.is_two_qubit() && gate_type.is_unitary_gate()
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
                Self::handle_before_gate(qubits, ctx)
            }
            NoiseEvent::AfterGate {
                gate_type,
                qubits,
                angles,
                ..
            } => {
                // Skip noise for noiseless gates
                if ctx.is_noiseless(*gate_type) {
                    return NoiseResponse::None;
                }
                self.handle_after_gate(*gate_type, qubits, angles, ctx, rng)
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
                if !gate_type.is_two_qubit() || !gate_type.is_unitary_gate() {
                    return None;
                }
                if ctx.is_noiseless(*gate_type) {
                    return Some(NoiseResponse::None);
                }
                Some(Self::handle_before_gate(qubits, ctx))
            }
            NoiseEvent::AfterGate {
                gate_type,
                qubits,
                angles,
                ..
            } => {
                if !gate_type.is_two_qubit() || !gate_type.is_unitary_gate() {
                    return None;
                }
                if ctx.is_noiseless(*gate_type) {
                    return Some(NoiseResponse::None);
                }
                Some(self.handle_after_gate(*gate_type, qubits, angles, ctx, rng))
            }
            _ => None,
        }
    }

    fn name(&self) -> &'static str {
        "TwoQubitChannel"
    }

    fn priority(&self) -> i32 {
        10
    }

    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

impl TwoQubitChannel {
    /// Handle `BeforeGate` - skip if any qubit is leaked.
    fn handle_before_gate(qubits: &[pecos_core::QubitId], ctx: &NoiseContext) -> NoiseResponse {
        // If any qubit is leaked, skip the gate
        // Uses optimized any_leaked which has O(1) fast path when leaked_count == 0
        if ctx.any_leaked(qubits) {
            return NoiseResponse::SkipGate;
        }
        NoiseResponse::None
    }

    /// Handle `AfterGate` - apply two-qubit Pauli errors.
    fn handle_after_gate(
        &self,
        gate_type: GateType,
        qubits: &[pecos_core::QubitId],
        angles: &[Angle64],
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse {
        if qubits.len() < 2 {
            return NoiseResponse::None;
        }

        let qubit0 = qubits[0];
        let qubit1 = qubits[1];

        // Fast path: if no qubits are leaked, skip leakage-related checks entirely
        let has_leakage = ctx.leaked_count() > 0;

        // Handle seepage for leaked qubits (using precomputed threshold)
        let mut unleaked = SmallVec::new();
        if has_leakage {
            if ctx.is_leaked(qubit0)
                && self.seepage_probability > 0.0
                && rng.check_probability(self.seepage_threshold)
            {
                unleaked.push(qubit0);
            }
            if ctx.is_leaked(qubit1)
                && self.seepage_probability > 0.0
                && rng.check_probability(self.seepage_threshold)
            {
                unleaked.push(qubit1);
            }

            // Skip error if either qubit is leaked
            if ctx.is_leaked(qubit0) || ctx.is_leaked(qubit1) {
                if unleaked.is_empty() {
                    return NoiseResponse::None;
                }
                return NoiseResponse::MarkUnleaked(unleaked);
            }
        }

        // Get angle-dependent error probability
        let p = self.effective_probability(gate_type, angles);

        if rng.random::<f64>() >= p {
            if unleaked.is_empty() {
                return NoiseResponse::None;
            }
            return NoiseResponse::MarkUnleaked(unleaked);
        }

        // Apply error
        let mut response = NoiseResponse::None;
        let mut gates = SmallVec::new();
        let mut leaked = SmallVec::new();

        // Determine if this is an emission or Pauli error (using precomputed threshold)
        if self.emission_ratio > 0.0 && rng.check_probability(self.emission_threshold) {
            // Emission REPLACES the gate: undo it (apply its dagger) so the net
            // effect is the gate removed, then apply the emission error. (Matches
            // engines' apply_tq_faults, which drops the original two-qubit gate
            // on a spontaneous-emission fault.)
            let original = GateCommand::with_angles(
                gate_type,
                smallvec::smallvec![qubit0, qubit1],
                angles.iter().copied().collect::<SmallVec<[Angle64; 2]>>(),
            );
            if let Some(dagger) = original.dagger() {
                gates.push(dagger);
            }
            // Emission error - sample from emission weights
            let idx = self.emission_weights.sample(rng.random::<f64>());
            let result = TwoQubitEmissionWeights::get_result(idx);

            // Apply Pauli gates if any
            if let Some(pauli) = result.first {
                gates.push(GateCommand::new(pauli, smallvec::smallvec![qubit0]));
            }
            if let Some(pauli) = result.second {
                gates.push(GateCommand::new(pauli, smallvec::smallvec![qubit1]));
            }

            // Track leakage
            if result.first_leaked {
                leaked.push(qubit0);
            }
            if result.second_leaked {
                leaked.push(qubit1);
            }
        } else {
            // Pauli error - sample from pauli weights
            let idx = self.pauli_weights.sample(rng.random::<f64>());
            let (pauli0, pauli1) = TwoQubitPauliWeights::get_paulis(idx);

            if pauli0 != GateType::I {
                gates.push(GateCommand::new(pauli0, smallvec::smallvec![qubit0]));
            }
            if pauli1 != GateType::I {
                gates.push(GateCommand::new(pauli1, smallvec::smallvec![qubit1]));
            }
        }

        if !gates.is_empty() {
            response = response.combine(NoiseResponse::inject_gates(gates));
        }
        if !leaked.is_empty() {
            response = response.combine(NoiseResponse::MarkLeaked(leaked));
        }
        if !unleaked.is_empty() {
            response = response.combine(NoiseResponse::MarkUnleaked(unleaked));
        }

        // Apply idle noise after the gate (memory sweeping, using precomputed threshold)
        if self.idle_rate > 0.0 {
            let mut idle_gates = SmallVec::new();
            for &qubit in &[qubit0, qubit1] {
                if !ctx.is_leaked(qubit) && rng.check_probability(self.idle_threshold) {
                    idle_gates.push(GateCommand::new(GateType::Z, smallvec::smallvec![qubit]));
                }
            }
            if !idle_gates.is_empty() {
                response = response.combine(NoiseResponse::inject_gates(idle_gates));
            }
        }

        response
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::QubitId;

    #[test]
    fn test_depolarizing_channel() {
        let channel = TwoQubitChannel::depolarizing(1.0);

        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
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
    fn test_no_error_on_single_qubit_gate() {
        let channel = TwoQubitChannel::depolarizing(1.0);

        let qubits = [QubitId(0)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        assert!(!channel.responds_to(&event));
    }

    #[test]
    fn test_angle_scaling() {
        let linear = AngleScaling::linear();
        assert!((linear.scale(Angle64::ZERO) - 0.0).abs() < 1e-10);
        // HALF_TURN = pi radians, normalized by pi = 1.0
        assert!((linear.scale(Angle64::HALF_TURN) - 1.0).abs() < 1e-10);
        // QUARTER_TURN = pi/2 radians, normalized by pi = 0.5
        assert!((linear.scale(Angle64::QUARTER_TURN) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_angle_dependent_probability() {
        let channel = TwoQubitChannel::depolarizing(0.1).with_angle_scaling(AngleScaling::linear());

        // For RZZ with angle pi, scaling = 1.0 (normalized)
        let p = channel.effective_probability(GateType::RZZ, &[Angle64::HALF_TURN]);
        assert!((p - 0.1).abs() < 1e-10);

        // For angle pi/2, scaling = 0.5
        let p = channel.effective_probability(GateType::RZZ, &[Angle64::QUARTER_TURN]);
        assert!((p - 0.05).abs() < 1e-10);
    }

    #[test]
    fn test_asymmetric_angle_scaling() {
        // Asymmetric: different scaling for positive vs negative angles
        // Formula: offset + linear * |theta/pi| + scale * |theta/pi|^power
        let scaling = AngleScaling::asymmetric(
            0.1, 2.0, 0.0, // negative: 0.1 + 2.0 * |theta/pi| + 0.0 * |theta/pi|^1
            0.2, 1.0, 0.0, // positive: 0.2 + 1.0 * |theta/pi| + 0.0 * |theta/pi|^1
            1.0, // power = 1 (linear)
        );

        // For positive pi/2 (normalized = 0.5): 0.2 + 1.0 * 0.5 + 0 = 0.7
        assert!((scaling.scale(Angle64::QUARTER_TURN) - 0.7).abs() < 1e-10);

        // For negative pi/2: 0.1 + 2.0 * 0.5 + 0 = 1.1
        // Note: Angle64 normalizes angles, so -pi/2 becomes 3pi/2 internally,
        // but to_radians_signed() converts back to -pi/2
        let neg_quarter = Angle64::from_radians(-std::f64::consts::FRAC_PI_2);
        assert!((scaling.scale(neg_quarter) - 1.1).abs() < 1e-10);

        // For zero: average of offsets = (0.1 + 0.2) / 2 = 0.15
        assert!((scaling.scale(Angle64::ZERO) - 0.15).abs() < 1e-10);

        // For positive pi (normalized = 1.0): 0.2 + 1.0 * 1.0 + 0 = 1.2
        assert!((scaling.scale(Angle64::HALF_TURN) - 1.2).abs() < 1e-10);
    }

    #[test]
    fn test_polynomial_angle_scaling() {
        // Test the full polynomial formula: a + b*|θ/π| + c*|θ/π|^d
        // Example: 0.1 + 0.2*|θ/π| + 0.5*|θ/π|^2
        let scaling = AngleScaling::polynomial(0.1, 0.2, 0.5, 2.0);

        // For θ = 0: just the constant = 0.1
        assert!((scaling.scale(Angle64::ZERO) - 0.1).abs() < 1e-10);

        // For θ = π/2 (normalized = 0.5):
        // 0.1 + 0.2 * 0.5 + 0.5 * 0.5^2 = 0.1 + 0.1 + 0.125 = 0.325
        assert!((scaling.scale(Angle64::QUARTER_TURN) - 0.325).abs() < 1e-10);

        // For θ = π (normalized = 1.0):
        // 0.1 + 0.2 * 1.0 + 0.5 * 1.0^2 = 0.1 + 0.2 + 0.5 = 0.8
        assert!((scaling.scale(Angle64::HALF_TURN) - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_from_general_params() {
        // Test compatibility with GeneralNoiseModel parameters
        let scaling = AngleScaling::from_general_params(0.01, 0.05, 0.1, 2.0);

        // Same as polynomial(0.01, 0.05, 0.1, 2.0)
        // For θ = π: 0.01 + 0.05 * 1.0 + 0.1 * 1.0^2 = 0.16
        assert!((scaling.scale(Angle64::HALF_TURN) - 0.16).abs() < 1e-10);
    }

    #[test]
    fn test_skip_gate_for_leaked_qubit() {
        let channel = TwoQubitChannel::depolarizing(1.0);

        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];
        let event = NoiseEvent::BeforeGate {
            gate_type: GateType::CX,
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
    fn test_biased_pauli_weights() {
        // Create a ZZ-biased channel (100% ZZ errors)
        let mut weights = [0.0; 15];
        weights[14] = 1.0; // ZZ only
        let channel = TwoQubitChannel::depolarizing(1.0)
            .with_pauli_weights(TwoQubitPauliWeights::custom(weights));

        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let response = channel.apply(&event, &mut ctx, &mut rng);

        // Should always produce ZZ error (Z on both qubits)
        if let NoiseResponse::InjectGates(gates) = response {
            assert_eq!(gates.len(), 2);
            assert_eq!(gates[0].gate_type, GateType::Z);
            assert_eq!(gates[0].qubits[0], QubitId(0));
            assert_eq!(gates[1].gate_type, GateType::Z);
            assert_eq!(gates[1].qubits[0], QubitId(1));
        } else {
            panic!("Expected InjectGates response");
        }
    }

    // ========================================================================
    // AngleScaling Edge Cases
    // ========================================================================

    #[test]
    fn test_angle_scaling_negative_angles() {
        // Test that negative angles use neg_* coefficients
        let scaling = AngleScaling::asymmetric(
            0.1, 0.5, 0.0, // neg: 0.1 + 0.5 * |θ/π|
            0.2, 1.0, 0.0, // pos: 0.2 + 1.0 * |θ/π|
            1.0,
        );

        // Negative π/2: 0.1 + 0.5 * 0.5 = 0.35
        let neg_quarter = Angle64::from_radians(-std::f64::consts::FRAC_PI_2);
        assert!((scaling.scale(neg_quarter) - 0.35).abs() < 1e-10);

        // Positive π/2: 0.2 + 1.0 * 0.5 = 0.7
        assert!((scaling.scale(Angle64::QUARTER_TURN) - 0.7).abs() < 1e-10);

        // Negative 3π/4: 0.1 + 0.5 * 0.75 = 0.475
        // (Using 3π/4 instead of π since -π and +π are equivalent)
        let neg_three_quarter_pi = Angle64::from_radians(-3.0 * std::f64::consts::FRAC_PI_4);
        assert!((scaling.scale(neg_three_quarter_pi) - 0.475).abs() < 1e-10);
    }

    #[test]
    fn test_angle_scaling_full_turn() {
        // Full turn (2π) should normalize to 0 in signed radians
        let scaling = AngleScaling::linear();

        // Full turn wraps to 0
        let full = Angle64::FULL_TURN;
        assert!((scaling.scale(full) - 0.0).abs() < 1e-10);

        // 3/4 turn = 3π/2 radians, signed = -π/2, normalized = 0.5
        let three_quarter = Angle64::from_radians(3.0 * std::f64::consts::FRAC_PI_2);
        assert!((scaling.scale(three_quarter) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_angle_scaling_mixed_polynomial() {
        // Test with all terms: constant + linear + power
        // 0.05 + 0.1*|θ/π| + 0.2*|θ/π|^3
        let scaling = AngleScaling::polynomial(0.05, 0.1, 0.2, 3.0);

        // θ = 0: just constant = 0.05
        assert!((scaling.scale(Angle64::ZERO) - 0.05).abs() < 1e-10);

        // θ = π/2 (norm = 0.5):
        // 0.05 + 0.1 * 0.5 + 0.2 * 0.5^3 = 0.05 + 0.05 + 0.025 = 0.125
        assert!((scaling.scale(Angle64::QUARTER_TURN) - 0.125).abs() < 1e-10);

        // θ = π (norm = 1.0):
        // 0.05 + 0.1 * 1.0 + 0.2 * 1.0^3 = 0.05 + 0.1 + 0.2 = 0.35
        assert!((scaling.scale(Angle64::HALF_TURN) - 0.35).abs() < 1e-10);
    }

    #[test]
    fn test_angle_scaling_quadratic_vs_polynomial() {
        // quadratic() should be equivalent to polynomial(0, 0, 1, 2)
        let quadratic = AngleScaling::quadratic();
        let polynomial = AngleScaling::polynomial(0.0, 0.0, 1.0, 2.0);

        for angle in [
            Angle64::ZERO,
            Angle64::QUARTER_TURN,
            Angle64::HALF_TURN,
            Angle64::from_radians(std::f64::consts::FRAC_PI_4),
        ] {
            let q_val = quadratic.scale(angle);
            let p_val = polynomial.scale(angle);
            assert!(
                (q_val - p_val).abs() < 1e-10,
                "Mismatch at {angle:?}: quadratic={q_val}, polynomial={p_val}"
            );
        }
    }

    #[test]
    fn test_angle_scaling_linear_vs_polynomial() {
        // linear() should be equivalent to polynomial(0, 1, 0, 1)
        let linear = AngleScaling::linear();
        let polynomial = AngleScaling::polynomial(0.0, 1.0, 0.0, 1.0);

        for angle in [
            Angle64::ZERO,
            Angle64::QUARTER_TURN,
            Angle64::HALF_TURN,
            Angle64::from_radians(std::f64::consts::FRAC_PI_4),
        ] {
            let l_val = linear.scale(angle);
            let p_val = polynomial.scale(angle);
            assert!(
                (l_val - p_val).abs() < 1e-10,
                "Mismatch at {angle:?}: linear={l_val}, polynomial={p_val}"
            );
        }
    }

    #[test]
    fn test_angle_scaling_constant_vs_polynomial() {
        // constant() should be equivalent to polynomial(1, 0, 0, 1)
        let constant = AngleScaling::constant();
        let polynomial = AngleScaling::polynomial(1.0, 0.0, 0.0, 1.0);

        for angle in [
            Angle64::ZERO,
            Angle64::QUARTER_TURN,
            Angle64::HALF_TURN,
            Angle64::from_radians(-std::f64::consts::FRAC_PI_2),
        ] {
            let c_val = constant.scale(angle);
            let p_val = polynomial.scale(angle);
            assert!(
                (c_val - p_val).abs() < 1e-10,
                "Mismatch at {angle:?}: constant={c_val}, polynomial={p_val}"
            );
        }
    }

    #[test]
    fn test_angle_scaling_builder_methods() {
        // Test the builder methods
        let scaling = AngleScaling::constant()
            .with_offset(0.1)
            .with_linear(0.2)
            .with_scale(0.3)
            .with_power(2.0);

        // θ = π (norm = 1.0):
        // 0.1 + 0.2 * 1.0 + 0.3 * 1.0^2 = 0.6
        assert!((scaling.scale(Angle64::HALF_TURN) - 0.6).abs() < 1e-10);

        // θ = π/2 (norm = 0.5):
        // 0.1 + 0.2 * 0.5 + 0.3 * 0.25 = 0.1 + 0.1 + 0.075 = 0.275
        assert!((scaling.scale(Angle64::QUARTER_TURN) - 0.275).abs() < 1e-10);
    }

    #[test]
    fn test_angle_scaling_small_angles() {
        // Test behavior with very small angles
        let scaling = AngleScaling::polynomial(0.01, 0.1, 1.0, 2.0);

        // Very small angle: should be dominated by constant term
        let tiny = Angle64::from_radians(0.001);
        let result = scaling.scale(tiny);
        // 0.01 + 0.1 * (0.001/π) + 1.0 * (0.001/π)^2 ≈ 0.01
        assert!(result > 0.009 && result < 0.011);
    }
}
