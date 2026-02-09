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

//! # pecos-neo
//!
//! Composable quantum simulation with event-driven noise modeling.
//!
//! This crate provides a composable approach to quantum simulation:
//!
//! - **Typed Commands**: [`GateCommand`] and [`CommandQueue`] replacing `ByteMessage`
//! - **Composable Noise**: Event-driven channels that can be freely combined
//! - **Plugin System**: Bevy-inspired architecture for bundling functionality
//! - **Simple Runner**: Direct simulator execution via [`ShotRunner`]
//!
//! ## Architecture
//!
//! The key insight is **composition over configuration**. Instead of a monolithic
//! noise model with dozens of parameters, you compose small, focused channels.
//!
//! ## Usage Patterns
//!
//! ### 1. Direct Composition (Most Flexible)
//!
//! ```
//! use pecos_neo::prelude::*;
//! use pecos_neo::noise::plugins::CorePlugin;
//!
//! let noise = ComposableNoiseModel::new()
//!     .add_plugin(CorePlugin)
//!     .add_channel(SingleQubitChannel::depolarizing(0.001))
//!     .add_channel(TwoQubitChannel::depolarizing(0.01))
//!     .add_channel(MeasurementChannel::asymmetric(0.02, 0.03));
//! ```
//!
//! ### 2. Convenience Builders (Familiar API)
//!
//! ```
//! use pecos_neo::noise::GeneralNoiseModelBuilder;
//!
//! let noise = GeneralNoiseModelBuilder::new()
//!     .with_p1(0.001)
//!     .with_p2(0.01)
//!     .with_p_meas(0.02, 0.03)
//!     .build();
//! ```
//!
//! ### 3. Mixed Approach (Best of Both)
//!
//! ```
//! use pecos_neo::prelude::*;
//! use pecos_neo::noise::GeneralNoiseModelBuilder;
//!
//! let noise = GeneralNoiseModelBuilder::new()
//!     .with_p1(0.001)
//!     .with_p2(0.01)
//!     .build()
//!     .add_channel(CrosstalkChannel::new()
//!         .with_global_rate(0.001));
//! ```
//!
//! ## Running Simulations
//!
//! ```
//! use pecos_neo::prelude::*;
//! use pecos_qsim::SparseStab;
//!
//! // Build a Bell state circuit
//! let commands = CommandBuilder::new()
//!     .prep(0)
//!     .prep(1)
//!     .h(0)
//!     .cx(0, 1)
//!     .measure(0)
//!     .measure(1)
//!     .build();
//!
//! // Run without noise
//! let mut runner = ShotRunner::new(SparseStab::new(2)).with_seed(42);
//! let outcomes = runner.execute(&commands);
//!
//! // Outcomes are correlated (Bell state)
//! let o0 = outcomes.get_bit(QubitId(0)).unwrap();
//! let o1 = outcomes.get_bit(QubitId(1)).unwrap();
//! assert_eq!(o0, o1);
//! ```
//!
//! ## With Noise
//!
//! ```
//! use pecos_neo::prelude::*;
//! use pecos_qsim::SparseStab;
//!
//! let commands = CommandBuilder::new()
//!     .prep(0)
//!     .h(0)
//!     .measure(0)
//!     .build();
//!
//! // Add depolarizing noise
//! let noise = ComposableNoiseModel::new()
//!     .add_channel(SingleQubitChannel::depolarizing(0.01))
//!     .add_channel(MeasurementChannel::symmetric(0.005));
//!
//! let mut runner = ShotRunner::new(SparseStab::new(1))
//!     .with_noise(noise)
//!     .with_seed(42);
//!
//! let outcomes = runner.execute(&commands);
//! ```
//!
//! ## With Rotation Gates (Universal Simulation)
//!
//! For simulators that support arbitrary rotation gates (like state vector simulators),
//! use `execute_all()` instead of `execute()`:
//!
//! ```
//! use pecos_neo::prelude::*;
//! use pecos_qsim::StateVec;
//!
//! let commands = CommandBuilder::new()
//!     .prep(0)
//!     .rx(0, Angle64::HALF_TURN)  // RX(pi) flips |0> to |1>
//!     .measure(0)
//!     .build();
//!
//! let mut runner = ShotRunner::new(StateVec::new(1)).with_seed(42);
//! let outcomes = runner.execute_all(&commands);
//!
//! assert!(outcomes.get_bit(QubitId(0)).unwrap());
//! ```

#![allow(clippy::module_inception)]

pub mod adapter;
pub mod circuit;
pub mod command;
pub mod ecs;
pub mod extended_runner;
pub mod extensible;
pub mod noise;
pub mod outcome;
pub mod program;
pub mod runner;
pub mod sampling;
pub mod tool;

// Re-export main types at crate root
pub use command::{CommandBuilder, CommandQueue, GateCommand, GateType};
pub use extensible::{
    GateId, GateSpec, GateCategory, GateRegistry, GateCanonicalizer, CanonicalForm,
    GateSupportSet, gates,
    AngleSnapper, SnapResult, SnapError, SnapPolicy,
    CircuitValidator, ValidationError, GateForValidation,
    CliffordValidator, CliffordTValidator, ExactAngleValidator,
    AllowListValidator, CompositeValidator,
    GateAdaptor, AdaptedGate, StandardAdaptor, CompositeAdaptor, CustomAdaptor,
    LiftedAdaptor, CompositeExtendedAdaptor,
    GateIdConversionError,
    CommandQueueValidation, snap_command_queue, is_clifford_circuit,
    is_clifford_gate_type, is_clifford_angle,
    // Extended operations for stabilizer measurements/preparations
    AdaptedOp, AdaptedSequence, AncillaRequirements, ResultId,
    PrepBasis, MeasBasis,
    ExtendedAdaptor, StabilizerAdaptor, StabilizerMeasurementAdaptor,
    StabilizerPreparationAdaptor, stabilizer_gates,
    // Arbitrary Pauli strings and operation builder
    Pauli, PauliString, StabilizerMeasurement, StabilizerPreparation,
    OpBuilder, Subcircuit, GateLibrary,
    // Gate definitions and execution
    GateDefinitions, GateDefinitionsBuilder, GateExecutor, NoNativeGates,
};
pub use noise::{
    ComposableNoiseModel, ContextObserver, EventHandler, GeneralNoiseModelBuilder, NoiseChannel,
    NoiseContext, NoiseEvent, NoiseModelConfig, NoisePlugin, NoiseResponse, PauliWeights,
    TwoQubitPauliWeights,
    context::QubitState,
    correlated::{CorrelatedNoiseChannel, CorrelationStats},
    crosstalk::CrosstalkChannel,
    gate_dependent::{GateDependentChannel, GateNoiseConfig},
    idle::IdleChannel,
    leakage::LeakageChannel,
    measurement::MeasurementChannel,
    preparation::PreparationChannel,
    single_qubit::SingleQubitChannel,
    two_qubit::TwoQubitChannel,
};
pub use outcome::{MeasurementOutcome, MeasurementOutcomes};
pub use program::{CommandSource, ConditionalProgram, ProgramResult, ProgramRunner, RepeatedProgram, StaticProgram};
pub use runner::ShotRunner;
pub use extended_runner::{ExtendedRunner, ExecutionError, GateOverrides, GateExecutorFn};

// Re-export adapter utilities (always available)
pub use adapter::{command_queue_to_gates, gate_to_command, gates_to_command_queue};

// Re-export ClassicalEngineAdapter when engines-adapter feature is enabled
#[cfg(feature = "engines-adapter")]
pub use adapter::{byte_message_to_command_queue, outcomes_to_byte_message, ClassicalEngineAdapter};

/// Prelude module for convenient imports.
///
/// # Example
///
/// ```
/// use pecos_neo::prelude::*;
/// ```
pub mod prelude {
    pub use crate::command::{CommandBuilder, CommandQueue, GateCommand, GateType};
    pub use crate::extensible::{
        GateId, GateSpec, GateCategory, GateRegistry, GateCanonicalizer, GateSupportSet, gates,
        AngleSnapper, SnapPolicy, CircuitValidator, CliffordValidator, ExactAngleValidator,
        GateAdaptor, StandardAdaptor,
        CommandQueueValidation, is_clifford_circuit,
        // Extended operations
        AdaptedOp, AdaptedSequence, ResultId, PrepBasis, MeasBasis,
        ExtendedAdaptor, StabilizerAdaptor, stabilizer_gates,
        Pauli, PauliString, StabilizerMeasurement, StabilizerPreparation, OpBuilder,
        // Gate definitions
        GateDefinitions, GateDefinitionsBuilder, GateExecutor,
    };
    pub use crate::noise::{
        ComposableNoiseModel, ContextObserver, EventHandler, GeneralNoiseModelBuilder,
        NoiseChannel, NoiseContext, NoiseEvent, NoiseModelConfig, NoisePlugin, NoiseResponse,
        PauliWeights, TwoQubitPauliWeights,
        context::QubitState,
        correlated::{CorrelatedNoiseChannel, CorrelationStats},
        crosstalk::CrosstalkChannel,
        gate_dependent::{GateDependentChannel, GateNoiseConfig},
        idle::IdleChannel,
        leakage::LeakageChannel,
        measurement::MeasurementChannel,
        plugins::{CorePlugin, DepolarizingPlugin, LeakagePlugin, MeasurementNoisePlugin},
        preparation::PreparationChannel,
        single_qubit::SingleQubitChannel,
        two_qubit::{AngleScaling, TwoQubitChannel},
    };
    pub use crate::outcome::{MeasurementOutcome, MeasurementOutcomes};
    pub use crate::runner::ShotRunner;
    pub use crate::extended_runner::{ExtendedRunner, ExecutionError, GateOverrides};

    // Re-export commonly used types from dependencies
    pub use pecos_core::{Angle64, QubitId};
}

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use pecos_qsim::SparseStab;

    #[test]
    fn test_prelude_usage() {
        let commands = CommandBuilder::new().prep(0).h(0).measure(0).build();

        let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);
        let outcomes = runner.execute(&commands);

        assert_eq!(outcomes.len(), 1);
    }

    #[test]
    fn test_bell_state_with_noise() {
        let commands = CommandBuilder::new()
            .prep(0)
            .prep(1)
            .h(0)
            .cx(0, 1)
            .measure(0)
            .measure(1)
            .build();

        // Very low noise to not disrupt Bell state correlation (legacy API)
        let noise = ComposableNoiseModel::new()
            .add_channel(SingleQubitChannel::depolarizing(0.0))
            .add_channel(TwoQubitChannel::depolarizing(0.0));

        let mut runner = ShotRunner::new(SparseStab::new(2))
            .with_noise(noise)
            .with_seed(42);

        let outcomes = runner.execute(&commands);

        // Bell state should still be correlated with zero noise
        let o0 = outcomes.get_bit(QubitId(0)).unwrap();
        let o1 = outcomes.get_bit(QubitId(1)).unwrap();
        assert_eq!(o0, o1);
    }

    #[test]
    fn test_plugin_based_noise_model() {
        let commands = CommandBuilder::new()
            .prep(0)
            .prep(1)
            .h(0)
            .cx(0, 1)
            .measure(0)
            .measure(1)
            .build();

        // Plugin-based noise model (recommended approach)
        let noise = ComposableNoiseModel::new()
            .add_plugin(CorePlugin)                        // State tracking
            .add_plugin(LeakagePlugin::new())              // Leakage handling
            .add_plugin(DepolarizingPlugin::new(0.0, 0.0)) // No noise
            .add_plugin(MeasurementNoisePlugin::symmetric(0.0));

        let mut runner = ShotRunner::new(SparseStab::new(2))
            .with_noise(noise)
            .with_seed(42);

        let outcomes = runner.execute(&commands);

        // Bell state should still be correlated
        let o0 = outcomes.get_bit(QubitId(0)).unwrap();
        let o1 = outcomes.get_bit(QubitId(1)).unwrap();
        assert_eq!(o0, o1);
    }

    #[test]
    fn test_multiple_shots() {
        let commands = CommandBuilder::new().prep(0).h(0).measure(0).build();

        let mut runner = ShotRunner::new(SparseStab::new(1)).with_seed(42);

        let mut count_0 = 0;
        let mut count_1 = 0;

        for _ in 0..100 {
            let outcomes = runner.run_shot(&commands);
            if outcomes.get_bit(QubitId(0)).unwrap() {
                count_1 += 1;
            } else {
                count_0 += 1;
            }
        }

        // Hadamard should give roughly 50/50 (allow for statistical fluctuation)
        assert!(
            count_0 > 30 && count_1 > 30,
            "Expected roughly 50/50 split, got {count_0}/{count_1}"
        );
    }
}
