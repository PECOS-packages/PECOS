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

//! Composite-based composable noise system.
//!
//! This module provides a primitive-based approach to building noise models
//! as decision trees. It complements the traditional standalone channel
//! implementations (`SingleQubitChannel`, `TwoQubitChannel`, etc.).
//!
//! ## Composite Channels vs Traditional Channels
//!
//! Both approaches integrate with `ComposableNoiseModel`. Choose based on your needs:
//!
//! **Use Composite Channels when:**
//! - You need complex conditional logic (leaked states, cross-qubit conditions)
//! - You want two-stage processing for correlated effects (partner depolarizing)
//! - You want to compose and reuse noise primitives
//! - You need custom branching or sampling behavior
//!
//! **Use Traditional Channels when:**
//! - You want simple, direct noise models
//! - Performance is critical (no primitive tree traversal)
//! - The built-in channel options suffice
//!
//! ## Key Concepts
//!
//! - **Primitives**: Composable building blocks (`Prob`, `When`, `Sample`, `Seq`, `TwoStage`)
//! - **Conditions**: State checks (`Leaked`, `PartnerFired`, `OutcomeIs`, custom)
//! - **Actions**: Terminal operations (`Pauli`, `Leak`, `Seep`, `AmplitudeDamping`, etc.)
//! - **Responses**: What to do (`InjectGates`, `SkipGate`, `FlipOutcome`, etc.)
//! - **Channels**: Integration with `ComposableNoiseModel` via `CompositeChannel`
//!
//! # Building Noise Decision Trees
//!
//! Use the `seq!` and `sample!` macros for heterogeneous compositions:
//!
//! ```
//! use pecos_neo::noise::composite::prelude::*;
//!
//! // Build single-qubit gate noise as a decision tree
//! let sq_noise = seq![
//!     skip_if_leaked(),  // Skip gate if qubit is leaked
//!     prob(0.01,         // 1% fault probability
//!         when_leaked(
//!             seep(),    // If leaked: seepage (return to computational basis)
//!             pauli()    // If not leaked: random Pauli error
//!         )
//!     ),
//! ];
//! ```
//!
//! # Integrating with `ComposableNoiseModel`
//!
//! Use `CompositeChannel` to integrate composite primitives with the noise model:
//!
//! ```
//! use pecos_neo::noise::composite::prelude::*;
//! use pecos_neo::noise::ComposableNoiseModel;
//!
//! // Create a composite-based noise channel
//! let gate_noise = seq![
//!     skip_if_leaked(),
//!     prob(0.01, pauli()),
//! ];
//! let channel = CompositeChannelBuilder::single_qubit("sq_depolarizing", gate_noise);
//!
//! // Add to composable noise model
//! let model = ComposableNoiseModel::new()
//!     .add_channel(channel);
//! ```
//!
//! # Using the `CompositeNoiseModelBuilder`
//!
//! For a simpler API similar to `GeneralNoiseModelBuilder`:
//!
//! ```
//! use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
//!
//! let model = CompositeNoiseModelBuilder::new()
//!     .with_p1(0.001)                 // Single-qubit gate error
//!     .with_p2(0.01)                  // Two-qubit gate error
//!     .with_p_meas(0.02, 0.03)        // Asymmetric measurement error
//!     .with_idle_noise(0.001, 0.01)   // T1 and T2 decay
//!     .with_leakage(0.001, 0.1)       // Leakage and seepage
//!     .build();
//! ```
//!
//! ## Advanced Builder Features
//!
//! The builder supports angle-dependent noise, leakage scaling, and custom Pauli weights:
//!
//! ```
//! use pecos_neo::noise::composite::prelude::*;
//!
//! let model = CompositeNoiseModelBuilder::new()
//!     // Angle-dependent two-qubit noise
//!     .with_p2(0.01)
//!     .with_p2_angle_scaling(AngleScaling::linear())
//!
//!     // Leakage scaling (0.5 = half become leakage, half depolarizing)
//!     .with_p1_emission_ratio(0.1)
//!     .with_leakage_scale(0.5)
//!
//!     // Custom Pauli weights for idle noise (Z-only for pure dephasing)
//!     .with_p_idle_linear(0.001)
//!     .with_p_idle_linear_weights(PauliWeights::custom(0.0, 0.0, 1.0))
//!
//!     .build();
//! ```
//!
//! # Idle Noise (T1/T2)
//!
//! Time-dependent idle noise can be modeled using `prob_linear` (T1-like)
//! and `prob_quadratic` (T2-like) primitives:
//!
//! ```
//! use pecos_neo::noise::composite::prelude::*;
//!
//! // T1 relaxation: probability grows linearly with time
//! let t1_channel = CompositeChannelBuilder::idle("t1",
//!     prob_linear(0.001, pauli())  // 0.001 per time unit
//! );
//!
//! // T2 dephasing: probability follows sin^2(rate * duration)
//! let t2_channel = CompositeChannelBuilder::idle("t2",
//!     prob_quadratic(0.01, inject_z())
//! );
//! ```
//!
//! # Dynamic Probability
//!
//! For angle-dependent or gate-type-dependent error rates:
//!
//! ```
//! use pecos_neo::noise::composite::prelude::*;
//!
//! // Angle-dependent two-qubit noise
//! let tq_noise = prob_fn(
//!     |gate| {
//!         gate.and_then(|g| g.angle())
//!             .map(|a| 0.01 * a.to_radians().abs())
//!             .unwrap_or(0.01)
//!     },
//!     pauli(),
//! );
//! ```

mod action;
pub mod batch;
pub mod batch_composite;
mod builder;
pub mod channel;
mod compiled;
mod condition;
mod primitive;
mod response;

// Re-export main types
pub use action::{
    CrosstalkAction, Emission, FlipOutcomeAction, ForceOutcomeAction, GateAction,
    IndependentEmissionWithPartnerDepolarize, Inject, InjectCoherentRZ, Leak,
    LeakedMeasurementAction, Nothing, PartnerDepolarize, Pauli, PauliWeights, RandomOutcome, Seep,
    SkipGate, TwoQubitEmission, TwoQubitEmissionWithPartnerDepolarize, TwoQubitPauli, Unleak,
};
pub use builder::CompositeNoiseModelBuilder;
pub use channel::{
    BatchCompositeChannel, CompositeChannel, CompositeChannelBuilder, CompositeCrosstalkChannel,
    CompositeEventFilter,
};
pub use compiled::{CompiledAction, CompiledCondition, CompiledPrimitive};
pub use condition::{
    Active, Always, AnyQubitLeaked, Condition, FnCondition, GateTypeIs, Leaked, Never, NotLeaked,
    OutcomeIs, PartnerLeaked,
};
pub use primitive::{
    BoxSample, BoxSeq, Primitive, Prob, ProbFn, ProbLinear, ProbQuadratic, Sample, Seq, SkipIf,
    TwoStage, When,
};
pub use response::CompositeResponse;

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::action::actions::*;
    pub use super::builder::CompositeNoiseModelBuilder;
    pub use super::channel::{
        BatchCompositeChannel, CompositeChannel, CompositeChannelBuilder,
        CompositeCrosstalkChannel, CompositeEventFilter,
    };
    pub use super::condition::conditions::*;
    pub use super::primitive::primitives::*;
    pub use super::{
        BoxSample, BoxSeq, CompositeResponse, GateAction, OutcomeIs, Pauli, PauliWeights,
        Primitive, ProbFn, ProbLinear, ProbQuadratic, TwoStage,
    };
    // Re-export GateInfo, IdleInfo, and AngleScaling for use in closures/builders
    pub use crate::noise::two_qubit::AngleScaling;
    pub use crate::noise::{GateInfo, IdleInfo};
    // Re-export macros
    pub use crate::{sample, seq};
}

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use crate::noise::NoiseContext;
    use pecos_core::QubitId;
    use pecos_random::PecosRng;

    /// Test that we can build a realistic single-qubit noise model.
    #[test]
    fn test_realistic_sq_noise() {
        let p1 = 0.01;
        let emission_ratio = 0.25;

        // This mirrors GeneralNoiseModel's single-qubit noise:
        // 1. Skip gate if leaked
        // 2. With probability p1:
        //    - If leaked: seepage
        //    - If not leaked: sample between emission and pauli

        let sq_noise = seq![
            skip_if_leaked(),
            prob(
                p1,
                when_leaked(
                    seep(),
                    sample![
                        (emission_ratio, leak()), // Simplified: emission -> leak
                        (1.0 - emission_ratio, pauli()),
                    ],
                ),
            ),
        ];

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Statistics for non-leaked qubit
        let mut no_fault = 0;
        let mut leaked_count = 0;
        let mut pauli_applied = 0;

        for _ in 0..10000 {
            ctx.mark_unleaked(QubitId(0));
            let response = sq_noise.apply(QubitId(0), &mut ctx, &mut rng);

            if response.is_none() {
                no_fault += 1;
            } else if response.causes_leak() {
                leaked_count += 1;
            } else if !response.collect_gates().is_empty() {
                pauli_applied += 1;
            }
        }

        let total = 10000.0;
        let no_fault_rate = f64::from(no_fault) / total;
        let leak_rate = f64::from(leaked_count) / total;
        let pauli_rate = f64::from(pauli_applied) / total;

        // Expected: ~99% no fault, ~0.25% leak, ~0.75% pauli
        assert!(
            (no_fault_rate - 0.99).abs() < 0.02,
            "no_fault_rate: {no_fault_rate}"
        );
        assert!((leak_rate - 0.0025).abs() < 0.01, "leak_rate: {leak_rate}");
        assert!(
            (pauli_rate - 0.0075).abs() < 0.01,
            "pauli_rate: {pauli_rate}"
        );
    }

    /// Test that leaked qubits get their gates skipped.
    #[test]
    fn test_leaked_qubit_skips_gate() {
        let sq_noise = seq![skip_if_leaked(), prob(1.0, pauli())];

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Mark qubit as leaked
        ctx.mark_leaked(QubitId(0));

        let response = sq_noise.apply(QubitId(0), &mut ctx, &mut rng);

        // Should skip, not apply pauli
        assert!(response.skips_gate());
        assert!(response.collect_gates().is_empty());
    }

    /// Test seepage for leaked qubits.
    #[test]
    fn test_leaked_qubit_seepage() {
        let sq_noise = prob(1.0, when_leaked(seep(), nothing()));

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Mark qubit as leaked
        ctx.mark_leaked(QubitId(0));
        assert!(ctx.is_leaked(QubitId(0)));

        // Apply noise - should seep (unleak)
        let _response = sq_noise.apply(QubitId(0), &mut ctx, &mut rng);

        // Should be unleaked now
        assert!(!ctx.is_leaked(QubitId(0)));
    }

    /// Integration test: `CompositeChannel` with `ComposableNoiseModel` and `CircuitRunner`.
    #[test]
    fn test_flow_channel_with_composable_model() {
        use crate::command::CommandBuilder;
        use crate::noise::ComposableNoiseModel;
        use crate::runner::CircuitRunner;
        use pecos_simulators::SparseStab;

        // Build a simple Hadamard circuit
        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        // Create a composite-based noise channel with 100% error rate for testing
        let sq_noise = prob(1.0, pauli());
        let channel = CompositeChannelBuilder::single_qubit("test_depolarizing", sq_noise);

        // Add to composable noise model
        let noise = ComposableNoiseModel::new().add_channel(channel);

        // Run with noise
        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);

        // Execute - the noise should apply after the H gate
        let _outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        // With 100% error rate and random Pauli, we expect ~1/3 each of X, Y, Z
        // This just verifies the integration works - detailed stats are in other tests
    }

    /// Test that composite channels can be combined with other noise channels.
    #[test]
    fn test_flow_channel_mixed_model() {
        use crate::command::CommandBuilder;
        use crate::noise::{ComposableNoiseModel, MeasurementChannel};
        use crate::runner::CircuitRunner;
        use pecos_simulators::SparseStab;

        // Build circuit
        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        // Composite-based gate noise
        let gate_noise = prob(0.0, pauli()); // No gate noise
        let flow_channel = CompositeChannelBuilder::single_qubit("flow_sq", gate_noise);

        // Traditional measurement noise channel
        let meas_channel = MeasurementChannel::symmetric(0.0); // No measurement noise

        // Combine both types of channels
        let noise = ComposableNoiseModel::new()
            .add_channel(flow_channel)
            .add_channel(meas_channel);

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);

        // Should work without errors
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        assert_eq!(outcomes.len(), 1);
    }

    /// Test outcome condition evaluation.
    #[test]
    fn test_outcome_condition() {
        use crate::noise::composite::Condition;
        use crate::noise::composite::condition::OutcomeIs;

        let mut ctx = NoiseContext::new();

        // No outcome set - should return false
        assert!(!OutcomeIs::zero().evaluate(QubitId(0), &ctx));
        assert!(!OutcomeIs::one().evaluate(QubitId(0), &ctx));

        // Set outcome to 0
        ctx.set_current_outcome(false);
        assert!(OutcomeIs::zero().evaluate(QubitId(0), &ctx));
        assert!(!OutcomeIs::one().evaluate(QubitId(0), &ctx));

        // Set outcome to 1
        ctx.set_current_outcome(true);
        assert!(!OutcomeIs::zero().evaluate(QubitId(0), &ctx));
        assert!(OutcomeIs::one().evaluate(QubitId(0), &ctx));

        // Clear outcome
        ctx.clear_current_outcome();
        assert!(!OutcomeIs::zero().evaluate(QubitId(0), &ctx));
    }

    /// Test `on_outcome` primitive.
    #[test]
    fn test_on_outcome_primitive() {
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Create: if outcome is 0, flip it
        let meas_noise = on_zero(flip_outcome());

        // Test with outcome 0 - should flip
        ctx.set_current_outcome(false);
        let response = meas_noise.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(response.flips_outcome());

        // Test with outcome 1 - should not flip
        ctx.set_current_outcome(true);
        let response = meas_noise.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(!response.flips_outcome());
    }

    /// Test building asymmetric measurement noise.
    #[test]
    fn test_asymmetric_measurement_noise() {
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Asymmetric noise: 2% error on 0, 5% error on 1
        // (using 100% here for deterministic testing)
        let meas_noise = seq![
            on_zero(prob(1.0, flip_outcome())), // Always flip 0 -> 1
            on_one(prob(0.0, flip_outcome())),  // Never flip 1 -> 0
        ];

        // Test with outcome 0 - should flip
        ctx.set_current_outcome(false);
        let response = meas_noise.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(response.flips_outcome());

        // Test with outcome 1 - should not flip
        ctx.set_current_outcome(true);
        let response = meas_noise.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(!response.flips_outcome());
    }

    /// Test `force_outcome` action.
    #[test]
    fn test_force_outcome_action() {
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Force outcome to 1 when leaked
        let leaked_meas_noise = when_leaked(force_one(), nothing());

        // Not leaked - should do nothing
        let response = leaked_meas_noise.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(response.is_none());

        // Leaked - should force to 1
        ctx.mark_leaked(QubitId(0));
        let response = leaked_meas_noise.apply(QubitId(0), &mut ctx, &mut rng);
        assert!(response.forces_outcome().is_some());
    }

    /// Test dynamic probability (`prob_fn`) through `CompositeChannel`.
    #[test]
    fn test_prob_fn_through_channel() {
        use crate::command::GateType;
        use crate::noise::NoiseChannel;
        use pecos_core::Angle64;

        // Create angle-dependent noise: error probability = angle / (2*pi)
        // Half turn (pi) -> 50%, quarter turn (pi/2) -> 25%
        let angle_dependent_noise = prob_fn(
            |gate| {
                gate.and_then(super::super::context::GateInfo::angle)
                    .map_or(0.0, |a| a.to_radians() / (2.0 * std::f64::consts::PI))
            },
            pauli(),
        );

        let channel = CompositeChannelBuilder::any_gate("angle_dependent", angle_dependent_noise);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Test with half turn angle (pi) -> expect ~50% error rate
        let qubits = [QubitId(0)];
        let angles = [Angle64::HALF_TURN];
        let event = crate::noise::NoiseEvent::AfterGate {
            gate_type: GateType::RZ,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        let mut error_count = 0;
        for _ in 0..1000 {
            let response = channel.apply(&event, &mut ctx, &mut rng);
            if !response.is_none() {
                error_count += 1;
            }
        }
        let error_rate = f64::from(error_count) / 1000.0;
        assert!(
            (error_rate - 0.5).abs() < 0.1,
            "Expected ~50% error rate for half turn, got {error_rate}"
        );

        // Test with quarter turn (pi/2) -> expect ~25% error rate
        let angles = [Angle64::QUARTER_TURN];
        let event = crate::noise::NoiseEvent::AfterGate {
            gate_type: GateType::RZ,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        error_count = 0;
        for _ in 0..1000 {
            let response = channel.apply(&event, &mut ctx, &mut rng);
            if !response.is_none() {
                error_count += 1;
            }
        }
        let error_rate = f64::from(error_count) / 1000.0;
        assert!(
            (error_rate - 0.25).abs() < 0.1,
            "Expected ~25% error rate for quarter turn, got {error_rate}"
        );
    }

    /// Test crosstalk channel integration with `ComposableNoiseModel`.
    #[test]
    fn test_crosstalk_with_composable_model() {
        use crate::command::CommandBuilder;
        use crate::noise::ComposableNoiseModel;
        use crate::runner::CircuitRunner;
        use pecos_simulators::SparseStab;

        // Create a circuit that measures qubit 0
        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .pz(&[2])
            .h(&[0])
            .mz(&[0])
            .build();

        // Create crosstalk channel: 100% chance to flip other qubits during measurement
        let crosstalk = CompositeCrosstalkChannel::new("test_crosstalk", inject_x())
            .responds_to_measurement()
            .global();

        let noise = ComposableNoiseModel::new().add_channel(crosstalk);

        let mut state = SparseStab::new(3);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);

        let _outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        // The test verifies the integration works without errors
        // Detailed behavior is tested in unit tests
    }

    /// Test gate-type dependent noise through `CompositeChannel`.
    #[test]
    fn test_gate_type_dependent_noise() {
        use crate::command::GateType;
        use crate::noise::NoiseChannel;

        // Create gate-type dependent noise:
        // - CX gates get 10% error (higher for two-qubit gates)
        // - H gates get 1% error
        // Using 100% and 0% for deterministic testing
        let gate_dependent_noise = prob_fn(
            |gate| gate.map_or(0.0, |g| if g.is_two_qubit() { 1.0 } else { 0.0 }),
            pauli(),
        );

        let channel = CompositeChannelBuilder::any_gate("gate_dependent", gate_dependent_noise);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Single-qubit gate (H) - should never trigger
        let qubits_1q = [QubitId(0)];
        let event_h = crate::noise::NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &qubits_1q,
            angles: &[],
            gate_id: None,
        };

        for _ in 0..100 {
            let response = channel.apply(&event_h, &mut ctx, &mut rng);
            assert!(response.is_none(), "H gate should have 0% error");
        }

        // Two-qubit gate (CX) - should always trigger
        let qubits_2q = [QubitId(0), QubitId(1)];
        let event_cx = crate::noise::NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits_2q,
            angles: &[],
            gate_id: None,
        };

        let mut error_count = 0;
        for _ in 0..100 {
            let response = channel.apply(&event_cx, &mut ctx, &mut rng);
            if !response.is_none() {
                error_count += 1;
            }
        }
        // CX affects both qubits, each should have 100% error
        assert_eq!(error_count, 100, "CX gate should have 100% error rate");
    }

    // ========================================================================
    // Comparison Tests (Flow vs Traditional Channels)
    // ========================================================================

    /// Compare composite-based single-qubit depolarizing to traditional `SingleQubitChannel`.
    #[test]
    fn test_flow_vs_traditional_single_qubit() {
        use crate::command::CommandBuilder;
        use crate::noise::{ComposableNoiseModel, PauliWeights, SingleQubitChannel};
        use crate::runner::CircuitRunner;
        use pecos_simulators::SparseStab;

        let p1 = 0.1; // 10% error rate for clear statistical signal

        // Build a circuit with repeated H gates
        let commands = CommandBuilder::new()
            .pz(&[0])
            .h(&[0])
            .h(&[0])
            .h(&[0])
            .h(&[0])
            .h(&[0])
            .mz(&[0])
            .build();

        // Run both with same seeds and count differences
        let shots = 1000;
        let mut traditional_errors = 0;
        let mut flow_errors = 0;

        for seed in 0..shots {
            // Traditional channel-based approach (recreate each iteration)
            let traditional_channel = SingleQubitChannel::new(
                p1,
                PauliWeights::uniform(),
                0.0,
                crate::noise::SingleQubitEmissionWeights::uniform(),
                0.0,
            );
            let traditional_noise = ComposableNoiseModel::new().add_channel(traditional_channel);

            let mut state_trad = SparseStab::new(1);
            let mut runner_trad = CircuitRunner::<SparseStab>::new()
                .with_noise(traditional_noise)
                .with_seed(seed);
            let outcomes_trad = runner_trad
                .apply_circuit(&mut state_trad, &commands)
                .unwrap();
            if outcomes_trad.get(QubitId(0)).is_none_or(|o| o.outcome) {
                traditional_errors += 1;
            }

            // Composite-based approach (recreate each iteration)
            let flow_noise = prob(p1, pauli());
            let flow_channel = CompositeChannelBuilder::single_qubit("flow_sq", flow_noise);
            let flow_noise_model = ComposableNoiseModel::new().add_channel(flow_channel);

            let mut state_flow = SparseStab::new(1);
            let mut runner_flow = CircuitRunner::<SparseStab>::new()
                .with_noise(flow_noise_model)
                .with_seed(seed);
            let outcomes_flow = runner_flow
                .apply_circuit(&mut state_flow, &commands)
                .unwrap();
            if outcomes_flow.get(QubitId(0)).is_none_or(|o| o.outcome) {
                flow_errors += 1;
            }
        }

        // Both should have similar error rates (within statistical tolerance)
        let trad_rate = f64::from(traditional_errors) / shots as f64;
        let flow_rate = f64::from(flow_errors) / shots as f64;

        // With 10% error rate per gate and 5 gates, we expect ~40% overall error rate
        // Allow 10% tolerance for statistical variation
        assert!(
            (trad_rate - flow_rate).abs() < 0.1,
            "Traditional rate {trad_rate:.3} vs Flow rate {flow_rate:.3} differ too much"
        );
    }

    /// Compare composite-based measurement noise to traditional `MeasurementChannel`.
    #[test]
    fn test_flow_vs_traditional_measurement() {
        use crate::command::CommandBuilder;
        use crate::noise::{ComposableNoiseModel, MeasurementChannel};
        use crate::runner::CircuitRunner;
        use pecos_simulators::SparseStab;

        let p_meas = 0.1; // 10% measurement error

        let commands = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

        // Run many shots and compare flip rates
        let shots = 1000;
        let mut traditional_flips = 0;
        let mut flow_flips = 0;

        for seed in 0..shots {
            // Traditional channel (recreate each iteration)
            let traditional_channel = MeasurementChannel::symmetric(p_meas);
            let traditional_noise = ComposableNoiseModel::new().add_channel(traditional_channel);

            let mut state_trad = SparseStab::new(1);
            let mut runner_trad = CircuitRunner::<SparseStab>::new()
                .with_noise(traditional_noise)
                .with_seed(seed);
            let outcomes_trad = runner_trad
                .apply_circuit(&mut state_trad, &commands)
                .unwrap();
            // Prep |0> and measure - should be 0 without error
            if outcomes_trad.get(QubitId(0)).is_some_and(|o| o.outcome) {
                traditional_flips += 1;
            }

            // Composite-based approach (recreate each iteration)
            let flow_meas_noise = prob(p_meas, flip_outcome());
            let flow_channel = CompositeChannel::new("flow_meas", flow_meas_noise)
                .with_filter(CompositeEventFilter::AfterMeasurement);
            let flow_noise_model = ComposableNoiseModel::new().add_channel(flow_channel);

            let mut state_flow = SparseStab::new(1);
            let mut runner_flow = CircuitRunner::<SparseStab>::new()
                .with_noise(flow_noise_model)
                .with_seed(seed);
            let outcomes_flow = runner_flow
                .apply_circuit(&mut state_flow, &commands)
                .unwrap();
            if outcomes_flow.get(QubitId(0)).is_some_and(|o| o.outcome) {
                flow_flips += 1;
            }
        }

        let trad_rate = f64::from(traditional_flips) / shots as f64;
        let flow_rate = f64::from(flow_flips) / shots as f64;

        // Both should be close to 10% flip rate
        assert!(
            (trad_rate - p_meas).abs() < 0.05,
            "Traditional flip rate {trad_rate:.3} far from expected {p_meas}"
        );
        assert!(
            (flow_rate - p_meas).abs() < 0.05,
            "Flow flip rate {flow_rate:.3} far from expected {p_meas}"
        );
        assert!(
            (trad_rate - flow_rate).abs() < 0.05,
            "Traditional rate {trad_rate:.3} vs Flow rate {flow_rate:.3} differ"
        );
    }

    /// Test that composite-based noise can replicate a complex `GeneralNoiseModel` configuration.
    #[test]
    fn test_flow_replicates_general_noise_model() {
        use crate::command::CommandBuilder;
        use crate::noise::{ComposableNoiseModel, GeneralNoiseModelBuilder};
        use crate::runner::CircuitRunner;
        use pecos_simulators::SparseStab;

        let p1 = 0.05;
        let p_meas = 0.02;

        // Build circuit: prep, several gates, measure
        let commands = CommandBuilder::new()
            .pz(&[0])
            .h(&[0])
            .h(&[0])
            .mz(&[0])
            .build();

        // Run both and compare error distributions
        let shots = 500;
        let mut traditional_ones = 0;
        let mut flow_ones = 0;

        for seed in 0..shots {
            // Traditional GeneralNoiseModelBuilder approach (recreate each iteration)
            let traditional_model = GeneralNoiseModelBuilder::new()
                .with_p1(p1)
                .with_p_meas_symmetric(p_meas)
                .build();

            let mut state_trad = SparseStab::new(1);
            let mut runner_trad = CircuitRunner::<SparseStab>::new()
                .with_noise(traditional_model)
                .with_seed(seed);
            if runner_trad
                .apply_circuit(&mut state_trad, &commands)
                .unwrap()
                .get(QubitId(0))
                .is_some_and(|o| o.outcome)
            {
                traditional_ones += 1;
            }

            // Composite-based equivalent (recreate each iteration)
            let sq_noise = prob(p1, pauli());
            let sq_channel = CompositeChannelBuilder::single_qubit("flow_sq", sq_noise);

            let meas_noise = prob(p_meas, flip_outcome());
            let meas_channel = CompositeChannel::new("flow_meas", meas_noise)
                .with_filter(CompositeEventFilter::AfterMeasurement);

            let flow_model = ComposableNoiseModel::new()
                .add_channel(sq_channel)
                .add_channel(meas_channel);

            let mut state_flow = SparseStab::new(1);
            let mut runner_flow = CircuitRunner::<SparseStab>::new()
                .with_noise(flow_model)
                .with_seed(seed);
            if runner_flow
                .apply_circuit(&mut state_flow, &commands)
                .unwrap()
                .get(QubitId(0))
                .is_some_and(|o| o.outcome)
            {
                flow_ones += 1;
            }
        }

        let trad_rate = f64::from(traditional_ones) / shots as f64;
        let flow_rate = f64::from(flow_ones) / shots as f64;

        // Both should produce similar overall error rates
        // The exact rates depend on the interaction of errors, but should be close
        assert!(
            (trad_rate - flow_rate).abs() < 0.1,
            "Traditional '1' rate {trad_rate:.3} vs Flow '1' rate {flow_rate:.3} differ too much"
        );
    }

    // ========================================================================
    // Idle Noise Tests
    // ========================================================================

    /// Test idle noise through `CompositeChannel` with `prob_linear` primitive.
    #[test]
    fn test_idle_noise_linear() {
        use crate::noise::NoiseChannel;
        use pecos_core::TimeUnits;

        // Create idle noise with linear time dependence: rate 0.1 per time unit
        let idle_noise = prob_linear(0.1, inject_z());
        let channel = CompositeChannelBuilder::idle("idle_linear", idle_noise);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Duration = 10 -> p = 0.1 * 10 = 1.0 (100%)
        let qubits = [QubitId(0)];
        let duration = TimeUnits::new(10);
        let event = crate::noise::NoiseEvent::IdleTime {
            qubits: &qubits,
            duration,
        };

        assert!(NoiseChannel::responds_to(&channel, &event));

        let mut error_count = 0;
        for _ in 0..100 {
            let response = channel.apply(&event, &mut ctx, &mut rng);
            if !response.is_none() {
                error_count += 1;
            }
        }
        assert!(
            error_count > 95,
            "Expected ~100% error rate for duration=10, got {error_count}%"
        );

        // Duration = 5 -> p = 0.1 * 5 = 0.5 (50%)
        let duration_half = TimeUnits::new(5);
        let event_half = crate::noise::NoiseEvent::IdleTime {
            qubits: &qubits,
            duration: duration_half,
        };

        error_count = 0;
        for _ in 0..1000 {
            let response = channel.apply(&event_half, &mut ctx, &mut rng);
            if !response.is_none() {
                error_count += 1;
            }
        }
        let rate = f64::from(error_count) / 1000.0;
        assert!(
            (rate - 0.5).abs() < 0.1,
            "Expected ~50% error rate for duration=5, got {rate}"
        );
    }

    /// Test idle noise through `CompositeChannel` with `prob_quadratic` primitive.
    #[test]
    fn test_idle_noise_quadratic() {
        use crate::noise::NoiseChannel;
        use pecos_core::TimeUnits;

        // Create idle noise with quadratic time dependence (T2-like dephasing)
        // Rate = pi/2 -> at duration 1, angle = pi/2, p = sin(pi/2)^2 = 1.0
        let idle_noise = prob_quadratic(std::f64::consts::FRAC_PI_2, inject_z());
        let channel = CompositeChannelBuilder::idle("idle_quadratic", idle_noise);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let qubits = [QubitId(0)];
        let duration = TimeUnits::new(1);
        let event = crate::noise::NoiseEvent::IdleTime {
            qubits: &qubits,
            duration,
        };

        assert!(NoiseChannel::responds_to(&channel, &event));

        let mut error_count = 0;
        for _ in 0..100 {
            let response = channel.apply(&event, &mut ctx, &mut rng);
            if !response.is_none() {
                error_count += 1;
            }
        }
        assert!(
            error_count > 95,
            "Expected ~100% error rate for T2 at duration=1, got {error_count}%"
        );
    }

    /// Test composite-based idle noise vs traditional `IdleChannel`.
    #[test]
    fn test_flow_vs_traditional_idle() {
        use crate::noise::{IdleChannel, NoiseChannel};
        use pecos_core::TimeUnits;

        let rate = 0.3; // 30% per time unit

        // Traditional IdleChannel
        let traditional = IdleChannel::linear(rate);

        // Composite-based equivalent
        let flow_noise = prob_linear(rate, inject_z());
        let flow_channel = CompositeChannelBuilder::idle("flow_idle", flow_noise);

        let mut ctx = NoiseContext::new();
        let mut rng_trad = PecosRng::seed_from_u64(42);
        let mut rng_flow = PecosRng::seed_from_u64(42);

        let qubits = [QubitId(0)];
        let duration = TimeUnits::new(1);
        let event = crate::noise::NoiseEvent::IdleTime {
            qubits: &qubits,
            duration,
        };

        ctx.mark_prepared(QubitId(0));

        // Run many times and compare rates
        let mut trad_count = 0;
        let mut flow_count = 0;
        for _ in 0..1000 {
            let trad_response = traditional.apply(&event, &mut ctx, &mut rng_trad);
            let flow_response = flow_channel.apply(&event, &mut ctx, &mut rng_flow);

            if !trad_response.is_none() {
                trad_count += 1;
            }
            if !flow_response.is_none() {
                flow_count += 1;
            }
        }

        let trad_rate = f64::from(trad_count) / 1000.0;
        let flow_rate = f64::from(flow_count) / 1000.0;

        // Both should be close to 30% error rate
        assert!(
            (trad_rate - rate).abs() < 0.1,
            "Traditional idle rate {trad_rate:.3} far from expected {rate}"
        );
        assert!(
            (flow_rate - rate).abs() < 0.1,
            "Flow idle rate {flow_rate:.3} far from expected {rate}"
        );
    }

    // ========================================================================
    // Crosstalk Action Tests
    // ========================================================================

    /// Test state-dependent crosstalk using `CrosstalkAction`.
    #[test]
    fn test_crosstalk_action_integration() {
        use super::Primitive;

        // Create crosstalk noise with state-dependent transitions
        let crosstalk_noise = crosstalk_with_leakage();

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Run many times and verify we get all three outcomes
        let mut no_change = 0;
        let mut flip = 0;
        let mut leak = 0;

        for _ in 0..3000 {
            let response = Primitive::apply(&crosstalk_noise, QubitId(0), &mut ctx, &mut rng);
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
        assert!(
            (f64::from(no_change) / total - 0.33).abs() < 0.1,
            "Expected ~33% no change"
        );
        assert!(
            (f64::from(flip) / total - 0.33).abs() < 0.1,
            "Expected ~33% flip"
        );
        assert!(
            (f64::from(leak) / total - 0.33).abs() < 0.1,
            "Expected ~33% leak"
        );
    }

    /// Test measuring leaked qubits with `RandomOutcome`.
    #[test]
    fn test_leaked_qubit_measurement() {
        // Build measurement noise for leaked qubits:
        // - If leaked: random outcome (biased towards 1)
        // - If not leaked: no effect
        let leaked_meas_noise = when_leaked(random_outcome(), nothing());

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Normal qubit - should do nothing
        for _ in 0..10 {
            let response = leaked_meas_noise.apply(QubitId(0), &mut ctx, &mut rng);
            assert!(response.is_none(), "Non-leaked qubit should have no effect");
        }

        // Leaked qubit - should force random outcome
        ctx.mark_leaked(QubitId(0));
        let mut ones = 0;
        for _ in 0..1000 {
            let response = leaked_meas_noise.apply(QubitId(0), &mut ctx, &mut rng);
            assert!(
                response.forces_outcome().is_some(),
                "Leaked qubit should force outcome"
            );
            if response.forces_outcome() == Some(true) {
                ones += 1;
            }
        }

        // With uniform random, should be ~50%
        let one_rate = f64::from(ones) / 1000.0;
        assert!(
            (one_rate - 0.5).abs() < 0.1,
            "Expected ~50% ones for uniform random, got {one_rate}"
        );
    }

    /// Test combining crosstalk transitions with probability.
    #[test]
    fn test_probabilistic_crosstalk() {
        // Create probabilistic crosstalk: 10% chance of state-dependent effect
        let crosstalk_noise = prob(0.1, crosstalk());

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let mut affected = 0;
        for _ in 0..1000 {
            let response = crosstalk_noise.apply(QubitId(0), &mut ctx, &mut rng);
            if !response.is_none() {
                affected += 1;
            }
        }

        // Should be roughly 10% affected (half of 10% probability since flip_only has 50% no change)
        let rate = f64::from(affected) / 1000.0;
        assert!(
            (rate - 0.05).abs() < 0.05,
            "Expected ~5% affected, got {rate}"
        );
    }

    // ========================================================================
    // Two-Stage Processing Tests
    // ========================================================================

    /// Test that two-stage primitives are processed correctly.
    ///
    /// This tests the full pipeline:
    /// 1. Stage 1 samples emission for each qubit
    /// 2. Stage 2 applies partner depolarizing based on cross-conditions
    #[test]
    fn test_two_stage_emission_with_partner_depolarize() {
        use crate::noise::NoiseChannel;

        // Create a two-stage primitive:
        // Stage 1: 100% emission probability (always fires)
        // Stage 2: If partner fired and I didn't, apply Pauli
        let noise = two_stage(
            prob(1.0, sample_emission()),                   // Stage 1: 100% emission
            when(partner_only_fired(), pauli(), nothing()), // Stage 2: partner depolarizing
        );

        let channel = CompositeChannel::new("two_stage_test", noise)
            .with_filter(CompositeEventFilter::TwoQubitGate);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        // Create a two-qubit gate event
        let qubits = [QubitId(0), QubitId(1)];
        let event = crate::noise::NoiseEvent::AfterGate {
            gate_type: crate::command::GateType::CX,
            qubits: &qubits,
            angles: &[],
            gate_id: None,
        };

        // With 100% emission probability, both qubits should emit/leak
        // Since both fired, partner_only_fired should be false for both
        // So no Pauli should be applied (just leakage)
        let _response = channel.apply(&event, &mut ctx, &mut rng);

        // Both qubits should be leaked
        assert!(ctx.is_leaked(QubitId(0)), "Qubit 0 should be leaked");
        assert!(ctx.is_leaked(QubitId(1)), "Qubit 1 should be leaked");

        // Response check: since both leaked, no Pauli gates needed
        // (we can't easily check the gates from NoiseResponse, but we verified leakage)
    }

    /// Test two-stage with partial emission (one qubit emits, other doesn't).
    #[test]
    fn test_two_stage_partial_emission() {
        use crate::noise::NoiseChannel;

        // Stage 1: Sample emission with 50% prob (we'll run many times to get both cases)
        // Stage 2: If partner fired and I didn't, apply Pauli
        let noise = two_stage(
            prob(0.5, sample_emission()),
            when(partner_only_fired(), pauli(), nothing()),
        );

        let channel = CompositeChannel::new("partial_test", noise)
            .with_filter(CompositeEventFilter::TwoQubitGate);

        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);

        let qubits = [QubitId(0), QubitId(1)];
        let event = crate::noise::NoiseEvent::AfterGate {
            gate_type: crate::command::GateType::CX,
            qubits: &qubits,
            angles: &[],
            gate_id: None,
        };

        // Run multiple times and count leakage patterns
        let mut both_leaked = 0;
        let mut one_leaked = 0;
        let mut neither_leaked = 0;

        for _ in 0..1000 {
            ctx.mark_unleaked(QubitId(0));
            ctx.mark_unleaked(QubitId(1));

            let _response = channel.apply(&event, &mut ctx, &mut rng);

            let q0_leaked = ctx.is_leaked(QubitId(0));
            let q1_leaked = ctx.is_leaked(QubitId(1));

            if q0_leaked && q1_leaked {
                both_leaked += 1;
            } else if q0_leaked || q1_leaked {
                one_leaked += 1;
            } else {
                neither_leaked += 1;
            }
        }

        // With 50% emission probability:
        // - ~25% both emit
        // - ~50% exactly one emits (partner should get Pauli)
        // - ~25% neither emits
        let total = 1000.0;
        let both_rate = f64::from(both_leaked) / total;
        let one_rate = f64::from(one_leaked) / total;
        let neither_rate = f64::from(neither_leaked) / total;

        // Allow wide tolerance for statistical tests
        assert!(
            (both_rate - 0.25).abs() < 0.1,
            "Expected ~25% both leaked, got {both_rate}"
        );
        assert!(
            (one_rate - 0.5).abs() < 0.15,
            "Expected ~50% one leaked, got {one_rate}"
        );
        assert!(
            (neither_rate - 0.25).abs() < 0.1,
            "Expected ~25% neither leaked, got {neither_rate}"
        );
    }

    // ========================================================================
    // Channel Adapter Tests
    // ========================================================================

    /// Test that `channel_action` wraps a traditional channel as a composite primitive.
    #[test]
    fn test_channel_action_wraps_traditional_channel() {
        use crate::command::GateType;
        use crate::noise::{NoiseChannel, SingleQubitChannel};

        let p1 = 0.5; // High error rate for clear signal

        // Create a traditional SingleQubitChannel
        let traditional_channel = SingleQubitChannel::new(
            p1,
            crate::noise::PauliWeights::uniform(),
            0.0,
            crate::noise::SingleQubitEmissionWeights::default(),
            0.0,
        );

        // Wrap it in a composite primitive and use within a decision tree
        let flow_noise = seq![skip_if_leaked(), channel_action(traditional_channel),];

        // Build a CompositeChannel from it
        let channel = CompositeChannelBuilder::single_qubit("adapted_channel", flow_noise);

        // Test that it responds appropriately
        let mut ctx = NoiseContext::new();
        ctx.set_current_gate(GateType::H, &[], 1);
        ctx.set_current_qubit_index(0, &[QubitId(0)]);
        let mut rng = PecosRng::seed_from_u64(42);

        // Run multiple trials
        let mut error_count = 0;
        for _ in 0..100 {
            let event = crate::noise::NoiseEvent::AfterGate {
                gate_type: GateType::H,
                qubits: &[QubitId(0)],
                angles: &[],
                gate_id: None,
            };
            let response = channel.apply(&event, &mut ctx, &mut rng);
            if !response.is_none() {
                error_count += 1;
            }
        }

        // With 50% error rate, we expect roughly 50 errors
        assert!(
            error_count > 30 && error_count < 70,
            "Expected ~50 errors, got {error_count}"
        );
    }

    /// Test `channel_action` in a conditional decision tree.
    #[test]
    fn test_channel_action_in_decision_tree() {
        use crate::command::GateType;
        use crate::noise::{NoiseChannel, SingleQubitChannel};

        // Traditional channel with 100% error rate (for deterministic testing)
        let always_error_channel = SingleQubitChannel::new(
            1.0,
            crate::noise::PauliWeights::uniform(),
            0.0,
            crate::noise::SingleQubitEmissionWeights::default(),
            0.0,
        );

        // Use channel_action in a conditional: only apply when not leaked
        let noise = when_leaked(
            nothing(),                            // Do nothing if leaked
            channel_action(always_error_channel), // Apply channel if not leaked
        );

        let channel = CompositeChannelBuilder::single_qubit("conditional_adapted", noise);

        let mut ctx = NoiseContext::new();
        ctx.set_current_gate(GateType::H, &[], 1);
        ctx.set_current_qubit_index(0, &[QubitId(0)]);
        let mut rng = PecosRng::seed_from_u64(42);

        // When not leaked: should apply channel and produce errors
        let event = crate::noise::NoiseEvent::AfterGate {
            gate_type: GateType::H,
            qubits: &[QubitId(0)],
            angles: &[],
            gate_id: None,
        };
        let _response = channel.apply(&event, &mut ctx, &mut rng);
        // The channel has 100% error rate, so we should see some response
        // (though the exact response depends on the Pauli sampled)

        // When leaked: should do nothing
        ctx.mark_leaked(QubitId(0));
        let response = channel.apply(&event, &mut ctx, &mut rng);
        // When leaked, the `when_leaked` condition is true, so we execute `nothing()`
        // and skip_if_leaked() isn't in this tree, so we get None
        assert!(
            response.is_none(),
            "Leaked qubit should not get errors from channel"
        );
    }

    // ========================================================================
    // Introspection Tests
    // ========================================================================

    /// Test that primitives have useful `describe()` output.
    #[test]
    fn test_primitive_describe() {
        // Simple action
        let action = pauli();
        assert_eq!(action.describe(), "pauli");

        // Prob with inner
        let prob_noise = prob(0.01, pauli());
        let desc = prob_noise.describe();
        assert!(desc.contains("0.01"), "Should contain probability: {desc}");
        assert!(desc.contains("pauli"), "Should contain inner: {desc}");

        // When with branches
        let when_noise = when_leaked(seep(), pauli());
        let desc = when_noise.describe();
        assert!(desc.contains("leaked"), "Should contain condition: {desc}");
    }

    /// Test that `describe_tree()` produces readable tree output.
    #[test]
    fn test_primitive_describe_tree() {
        use crate::noise::composite::Primitive;

        // Build a complex decision tree
        let noise = seq![
            skip_if_leaked(),
            prob(
                0.01,
                when_leaked(seep(), sample![(0.25, leak()), (0.75, pauli()),],)
            ),
        ];

        let tree = noise.describe_tree();

        // Should have tree structure
        assert!(tree.contains("Seq"), "Should have Seq node: {tree}");
        assert!(tree.contains("Prob"), "Should have Prob node: {tree}");
        assert!(tree.contains("When"), "Should have When node: {tree}");
        assert!(tree.contains("Sample"), "Should have Sample node: {tree}");
        assert!(tree.contains("skip_if"), "Should have skip_if: {tree}");
        assert!(tree.contains("seep"), "Should have seep: {tree}");
        assert!(tree.contains("pauli"), "Should have pauli: {tree}");
        assert!(tree.contains("leak"), "Should have leak: {tree}");

        // Print for manual inspection during development
        // println!("{}", tree);
    }

    /// Test `ComposableNoiseModel` introspection.
    #[test]
    fn test_model_introspection() {
        use crate::noise::ComposableNoiseModel;

        let channel1 =
            CompositeChannelBuilder::single_qubit("sq_depolarizing", prob(0.01, pauli()));
        let channel2 = CompositeChannelBuilder::after_measurement("meas_error", flip_outcome());

        let model = ComposableNoiseModel::new()
            .add_channel(channel1)
            .add_channel(channel2);

        // Check channel count
        assert_eq!(model.channel_count(), 2);

        // Check channel names
        let names = model.channel_names();
        assert!(names.contains(&"sq_depolarizing"));
        assert!(names.contains(&"meas_error"));

        // Check describe output
        let desc = model.describe();
        assert!(
            desc.contains("ComposableNoiseModel"),
            "Should have header: {desc}"
        );
        assert!(desc.contains("Channels: 2"), "Should show count: {desc}");
        assert!(
            desc.contains("sq_depolarizing"),
            "Should list channels: {desc}"
        );
    }
}
