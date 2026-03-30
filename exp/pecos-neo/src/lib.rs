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
//! - **Simple `CircuitRunner`**: Direct simulator execution via [`CircuitRunner`]
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
//! use pecos_simulators::SparseStab;
//!
//! // Build a Bell state circuit
//! let commands = CommandBuilder::new()
//!     .pz(&[0])
//!     .pz(&[1])
//!     .h(&[0])
//!     .cx(&[(0, 1)])
//!     .mz(&[0])
//!     .mz(&[1])
//!     .build();
//!
//! // Run without noise
//! let mut state = SparseStab::new(2);
//! let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
//! let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
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
//! use pecos_simulators::SparseStab;
//!
//! let commands = CommandBuilder::new()
//!     .pz(&[0])
//!     .h(&[0])
//!     .mz(&[0])
//!     .build();
//!
//! // Add depolarizing noise
//! let noise = ComposableNoiseModel::new()
//!     .add_channel(SingleQubitChannel::depolarizing(0.01))
//!     .add_channel(MeasurementChannel::symmetric(0.005));
//!
//! let mut state = SparseStab::new(1);
//! let mut runner = CircuitRunner::<SparseStab>::new()
//!     .with_noise(noise)
//!     .with_seed(42);
//!
//! let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
//! ```
//!
//! ## With Rotation Gates (Universal Simulation)
//!
//! For simulators that support arbitrary rotation gates (like state vector simulators),
//! use `CircuitRunner::rotations()`:
//!
//! ```
//! use pecos_neo::prelude::*;
//! use pecos_simulators::StateVec;
//!
//! let commands = CommandBuilder::new()
//!     .pz(&[0])
//!     .rx(&[0], Angle64::HALF_TURN)  // RX(pi) flips |0> to |1>
//!     .mz(&[0])
//!     .build();
//!
//! let mut state = StateVec::new(1);
//! let mut runner = CircuitRunner::<StateVec>::rotations().with_seed(42);
//! let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
//!
//! assert!(outcomes.get_bit(QubitId(0)).unwrap());
//! ```

#![allow(clippy::module_inception)]

pub mod adapter;
pub mod circuit;
pub mod command;
pub mod ecs;
pub mod engines;
pub mod extensible;
pub mod noise;
pub mod outcome;
pub mod program;
pub mod runner;
pub mod sampling;
pub mod tool;

// Re-export main types at crate root
pub use command::{CommandBuilder, CommandQueue, GateCommand, GateType};
pub use engines::{CommandQueueEngine, DagCircuitEngine, TickCircuitEngine};
pub use extensible::{
    AdaptedGate,
    // Extended operations for stabilizer measurements/preparations
    AdaptedOp,
    AdaptedSequence,
    AllowListValidator,
    AncillaRequirements,
    AngleSnapper,
    CanonicalForm,
    CircuitValidator,
    CliffordTValidator,
    CliffordValidator,
    CommandQueueValidation,
    CompositeAdaptor,
    CompositeExtendedAdaptor,
    CompositeValidator,
    CustomAdaptor,
    ExactAngleValidator,
    ExtendedAdaptor,
    GateAdaptor,
    GateCanonicalizer,
    GateCategory,
    // Gate definitions and execution
    GateDefinitions,
    GateDefinitionsBuilder,
    GateExecutor,
    GateForValidation,
    GateId,
    GateIdConversionError,
    GateLibrary,
    GateRegistry,
    GateSpec,
    GateSupportSet,
    LiftedAdaptor,
    MeasBasis,
    NoNativeGates,
    OpBuilder,
    // Arbitrary Pauli strings and operation builder
    Pauli,
    PauliString,
    PrepBasis,
    ResultId,
    SnapError,
    SnapPolicy,
    SnapResult,
    StabilizerAdaptor,
    StabilizerMeasurement,
    StabilizerMeasurementAdaptor,
    StabilizerPreparation,
    StabilizerPreparationAdaptor,
    StandardAdaptor,
    Subcircuit,
    ValidationError,
    gates,
    is_clifford_angle,
    is_clifford_circuit,
    is_clifford_gate_type,
    snap_command_queue,
    stabilizer_gates,
};
pub use noise::{
    ComposableNoiseModel, ContextObserver, EventHandler, GeneralNoiseModelBuilder, NoiseChannel,
    NoiseContext, NoiseEvent, NoiseModelConfig, NoisePlugin, NoiseResponse, PauliWeights,
    TwoQubitPauliWeights,
    context::QubitState,
    correlated::{CorrelatedNoiseChannel, CorrelationStats},
    crosstalk::CrosstalkChannel,
    gate_dependent::{GateDependentChannel, GateNoiseConfig},
    general_noise,
    idle::IdleChannel,
    leakage::LeakageChannel,
    measurement::MeasurementChannel,
    preparation::PreparationChannel,
    single_qubit::SingleQubitChannel,
    two_qubit::TwoQubitChannel,
};
pub use outcome::{MeasurementOutcome, MeasurementOutcomes, RegisterMap};
pub use program::{
    CommandSource, ConditionalProgram, DynProgramRunner, ProgramResult, ProgramRunner,
    RepeatedProgram, StaticProgram,
};
pub use runner::DispatchContext;
pub use runner::{CircuitRunner, EventHandlers, ExecutionError, GateExecutorFn, GateOverrides};

// Re-export adapter utilities (always available)
pub use adapter::{command_queue_to_gates, gate_to_command, gates_to_command_queue};

// Re-export ClassicalEngineAdapter when engines-adapter feature is enabled
#[cfg(feature = "engines-adapter")]
pub use adapter::{
    ClassicalEngineAdapter, byte_message_to_command_queue, outcomes_to_byte_message,
};

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
        // Extended operations
        AdaptedOp,
        AdaptedSequence,
        AngleSnapper,
        CircuitValidator,
        CliffordValidator,
        CommandQueueValidation,
        ExactAngleValidator,
        ExtendedAdaptor,
        GateAdaptor,
        GateCanonicalizer,
        GateCategory,
        // Gate definitions
        GateDefinitions,
        GateDefinitionsBuilder,
        GateExecutor,
        GateId,
        GateRegistry,
        GateSpec,
        GateSupportSet,
        MeasBasis,
        OpBuilder,
        Pauli,
        PauliString,
        PrepBasis,
        ResultId,
        SnapPolicy,
        StabilizerAdaptor,
        StabilizerMeasurement,
        StabilizerPreparation,
        StandardAdaptor,
        gates,
        is_clifford_circuit,
        stabilizer_gates,
    };
    pub use crate::noise::{
        ComposableNoiseModel, ContextObserver, EventHandler, GeneralNoiseModelBuilder,
        NoiseChannel, NoiseContext, NoiseEvent, NoiseModelConfig, NoisePlugin, NoiseResponse,
        PauliWeights, TwoQubitPauliWeights,
        context::QubitState,
        correlated::{CorrelatedNoiseChannel, CorrelationStats},
        crosstalk::CrosstalkChannel,
        gate_dependent::{GateDependentChannel, GateNoiseConfig},
        general_noise,
        idle::IdleChannel,
        leakage::LeakageChannel,
        measurement::MeasurementChannel,
        plugins::{CorePlugin, DepolarizingPlugin, LeakagePlugin, MeasurementNoisePlugin},
        preparation::PreparationChannel,
        single_qubit::SingleQubitChannel,
        two_qubit::{AngleScaling, TwoQubitChannel},
    };
    pub use crate::outcome::{MeasurementOutcome, MeasurementOutcomes};
    pub use crate::runner::DispatchContext;
    pub use crate::runner::{CircuitRunner, EventHandlers, ExecutionError, GateOverrides};

    // Re-export commonly used types from dependencies
    pub use pecos_core::{Angle64, QubitId};
}

#[cfg(test)]
mod tests {
    use super::prelude::*;
    use pecos_simulators::SparseStab;

    #[test]
    fn test_prelude_usage() {
        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        assert_eq!(outcomes.len(), 1);
    }

    #[test]
    fn test_bell_state_with_noise() {
        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .h(&[0])
            .cx(&[(0, 1)])
            .mz(&[0])
            .mz(&[1])
            .build();

        let noise = ComposableNoiseModel::new()
            .add_channel(SingleQubitChannel::depolarizing(0.0))
            .add_channel(TwoQubitChannel::depolarizing(0.0));

        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);

        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        let o0 = outcomes.get_bit(QubitId(0)).unwrap();
        let o1 = outcomes.get_bit(QubitId(1)).unwrap();
        assert_eq!(o0, o1);
    }

    #[test]
    fn test_plugin_based_noise_model() {
        let commands = CommandBuilder::new()
            .pz(&[0])
            .pz(&[1])
            .h(&[0])
            .cx(&[(0, 1)])
            .mz(&[0])
            .mz(&[1])
            .build();

        let noise = ComposableNoiseModel::new()
            .add_plugin(CorePlugin)
            .add_plugin(LeakagePlugin::new())
            .add_plugin(DepolarizingPlugin::new(0.0, 0.0))
            .add_plugin(MeasurementNoisePlugin::symmetric(0.0));

        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(42);

        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        let o0 = outcomes.get_bit(QubitId(0)).unwrap();
        let o1 = outcomes.get_bit(QubitId(1)).unwrap();
        assert_eq!(o0, o1);
    }

    #[test]
    fn test_multiple_shots() {
        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);

        let mut count_0 = 0;
        let mut count_1 = 0;

        for _ in 0..100 {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
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
