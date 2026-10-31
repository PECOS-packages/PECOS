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

//! Unified event-driven noise system.
//!
//! This module provides a composable noise model where all noise is event-driven.
//! You can work at whatever level of abstraction you need:
//!
//! - **Simple**: Set error rates and let the builder create appropriate channels
//! - **Composed**: Build decision trees using primitives (prob, when, seq, etc.)
//! - **Custom**: Implement your own channels for complete control
//!
//! Decision-tree primitives are available with the `composite-noise` Cargo feature.
//!
//! # Quick Start
//!
//! Import everything with the prelude:
//!
//! ```no_run
//! use pecos_neo::noise::prelude::*;
//!
//! // Simple: just set error rates
//! let simple = NoiseModelBuilder::new()
//!     .with_depolarizing(0.001, 0.01)
//!     .with_measurement_error(0.02)
//!     .build()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! Mixed approach (combine builder with custom channels):
//!
//! ```
//! # use pecos_neo::noise::prelude::*;
//! // Mixed: combine both approaches
//! let custom_channel = MeasurementChannel::symmetric(0.02)?;
//! let mixed = NoiseModelBuilder::new()
//!     .with_depolarizing(0.001, 0.01)
//!     .with_channel(custom_channel)
//!     .build()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Architecture
//!
//! ```text
//!                     ┌─────────────────────────────────────┐
//!                     │     Unified Event-Driven Noise      │
//!                     │          (NoiseChannel trait)       │
//!                     └─────────────────────────────────────┘
//!                                       │
//!                     ┌─────────────────┴─────────────────┐
//!                     │                                   │
//!             ┌───────▼───────┐               ┌──────────▼──────────┐
//!             │ Base Channels │               │  Composed Channels  │
//!             │  (atomic ops) │               │   (decision trees)  │
//!             └───────────────┘               └─────────────────────┘
//!                     │                                   │
//!          ┌──────────┼──────────┐              uses base channels
//!          │          │          │              as building blocks
//!     Depolarizing  Amplitude  Measurement            │
//!        Pauli      Damping     Flip                  │
//!          │          │          │                    │
//!          └──────────┴──────────┴────────────────────┘
//!                               │
//!                     ┌─────────▼─────────┐
//!                     │ ComposableNoiseModel │
//!                     │   (combines all)     │
//!                     └─────────────────────┘
//! ```
//!
//! # Key Concepts
//!
//! - **Events**: Notifications about operations (gates, measurements, etc.)
//! - **Channels**: Components that respond to events with noise injection
//! - **Composition**: Combining channels with logic (prob, when, seq, sample)
//! - **Context**: Shared state (leakage tracking, prepared qubits)
//!
//! # Event Flow
//!
//! 1. `BeforeGate` - can skip operand groups (for leaked qubits)
//! 2. `AfterGate` - can inject Pauli errors
//! 3. `AfterMeasurement` - can flip outcomes
//! 4. `AfterPreparation` - can inject bit-flip errors
//! 5. `IdleTime` - can apply T1/T2 decay

pub mod builder;
pub mod category_channel;
pub mod composer;
#[cfg(feature = "composite-noise")]
pub mod composite;
pub mod context;
pub mod correlated;
pub mod crosstalk;
pub mod gate_dependent;
pub mod gate_id_dependent;
pub mod general_builder;
pub mod idle;
pub mod introspection;
pub mod leakage;
pub mod measurement;
pub mod patterns;
pub mod per_gate_pauli;
pub mod plugin;
pub mod plugins;
pub mod prelude;
pub mod preparation;
pub mod single_qubit;
pub mod topology;
pub mod two_qubit;
pub mod validation;

#[cfg(test)]
mod event_kind_tests;

pub use builder::NoiseModelBuilder;
pub use category_channel::CategoryBasedChannel;
pub use composer::ComposableNoiseModel;
pub use context::{BitVec, GateInfo, IdleInfo, NoiseContext, QubitState};
pub use correlated::{CorrelatedNoiseChannel, CorrelationStats};
pub use crosstalk::CrosstalkChannel;
pub use gate_dependent::{GateDependentChannel, GateNoiseConfig};
pub use gate_id_dependent::{GateIdDependentChannel, GateIdNoiseConfig};
pub use general_builder::{GeneralNoiseModelBuilder, general_noise};
pub use idle::IdleChannel;
pub use leakage::LeakageChannel;
pub use measurement::{MeasurementChannel, MeasurementStateFlipChannel};
pub use per_gate_pauli::PerGatePauliChannel;
pub use plugin::{ContextObserver, EventHandler, NoiseModelConfig, NoisePlugin};
pub use preparation::PreparationChannel;
pub use single_qubit::SingleQubitChannel;
pub use two_qubit::{AngleScaling, TwoQubitChannel};

use self::validation::{ValidationResult, validate_distribution, validate_probability};
use crate::command::GateCommand;
use crate::command::GateType;
use crate::extensible::GateId;
use pecos_core::{Angle64, QubitId, Signal, TimeUnits};
use pecos_random::PecosRng;
use smallvec::SmallVec;
use std::any::{Any, TypeId};
use std::borrow::Cow;

macro_rules! declare_noise_event_kinds {
    ($($variant:ident),+ $(,)?) => {
        /// The payload-independent kind of a noise event.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum NoiseEventKind {
            $(
                #[doc = concat!("The kind of [`NoiseEvent::", stringify!($variant), "`].")]
                $variant,
            )+
        }

        impl NoiseEventKind {
            pub(crate) const COUNT: usize = [$(Self::$variant),+].len();
            pub(crate) const ALL: [Self; Self::COUNT] = [$(Self::$variant),+];

            pub(crate) const fn index(self) -> usize {
                // Implicit discriminants follow the declaration (and ALL) order.
                self as usize
            }
        }

        // Keep the list exhaustive even if the enum generation is later changed.
        const fn noise_event_kinds_are_exhaustive(kind: NoiseEventKind) {
            match kind {
                $(NoiseEventKind::$variant => {}),+
            }
        }

        const _: () = noise_event_kinds_are_exhaustive(NoiseEventKind::ALL[0]);
    };
}

declare_noise_event_kinds!(
    BeforeGate,
    AfterGate,
    BeforeMeasurement,
    AfterMeasurement,
    AfterPreparation,
    IdleTime,
    AfterReset,
    BeforeCircuit,
    AfterCircuit,
    BetweenLayers,
    Signal,
);

/// Construction-time event subscriptions for a noise channel or handler.
///
/// A declaration may include extra kinds, but must include every kind the
/// implementation can handle. The composer caches this set at insertion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EventKinds(u16);

impl EventKinds {
    /// Number of distinct noise event kinds.
    pub const COUNT: usize = NoiseEventKind::COUNT;
    /// Subscribe to every event kind (the conservative default).
    pub const ALL: Self = Self((1 << Self::COUNT) - 1);
    /// Subscribe to no event kinds.
    pub const NONE: Self = Self(0);

    /// Subscribe to a single event kind.
    #[must_use]
    pub const fn of(kind: NoiseEventKind) -> Self {
        Self(1 << kind.index())
    }

    /// Include another kind.
    #[must_use]
    pub const fn with(self, kind: NoiseEventKind) -> Self {
        self.union(Self::of(kind))
    }

    /// Check whether a kind is included.
    #[must_use]
    pub const fn contains(self, kind: NoiseEventKind) -> bool {
        self.0 & Self::of(kind).0 != 0
    }

    /// Combine two declarations.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

/// Events that can trigger noise in the simulation.
///
/// Noise channels subscribe to specific events and can inject errors
/// when those events occur.
#[derive(Debug, Clone)]
pub enum NoiseEvent<'a> {
    /// Emitted before a gate is applied.
    ///
    /// Channels can respond with `SkipGate(qubits)` to suppress operand groups
    /// containing the listed qubits (e.g., leaked qubits).
    BeforeGate {
        gate_type: GateType,
        qubits: &'a [QubitId],
        angles: &'a [Angle64],
        /// Optional gate ID for custom gate identification.
        /// This allows noise channels to apply per-custom-gate noise rates.
        /// `None` for standard gates or when gate ID tracking is disabled.
        gate_id: Option<GateId>,
    },

    /// Emitted after a gate is applied.
    ///
    /// Carries executed groups and groups suppressed while containing a leaked
    /// qubit, in original operand order. Leakage is checked immediately after
    /// before-gate dispatch. Other suppressed groups are absent; if that removes
    /// every operand from a non-empty command, no after-gate event is emitted.
    /// Absent groups receive no idle-after-two-qubit-gate noise and are not
    /// crosstalk sources. Originally empty operand slices retain their normal
    /// event behavior.
    ///
    /// Channels can respond with `InjectGates` to add Pauli errors.
    AfterGate {
        gate_type: GateType,
        qubits: &'a [QubitId],
        angles: &'a [Angle64],
        /// Optional gate ID for custom gate identification.
        /// This allows noise channels to apply per-custom-gate noise rates.
        /// `None` for standard gates or when gate ID tracking is disabled.
        gate_id: Option<GateId>,
    },

    /// Emitted before measurement.
    ///
    /// Used for pre-measurement effects like leaked qubit handling.
    BeforeMeasurement { qubits: &'a [QubitId] },

    /// Emitted after measurement results are available.
    ///
    /// Channels can respond with `FlipOutcomes` to apply readout errors.
    AfterMeasurement {
        qubits: &'a [QubitId],
        outcomes: &'a [bool],
    },

    /// Emitted after state preparation.
    ///
    /// Channels can respond with bit-flip errors or leakage.
    AfterPreparation { qubits: &'a [QubitId] },

    /// Emitted for idle time on qubits.
    ///
    /// Used for T1/T2 decay and dephasing.
    /// Duration is in abstract time units - interpretation is defined by config.
    IdleTime {
        qubits: &'a [QubitId],
        duration: TimeUnits,
    },

    /// Emitted after a mid-circuit reset operation.
    ///
    /// Reset operations prepare qubits to |0> during circuit execution.
    /// This event can be used for reset-specific noise like imperfect
    /// state preparation or leakage.
    AfterReset { qubits: &'a [QubitId] },

    /// Emitted at the start of circuit execution.
    ///
    /// Used for initialization noise or context setup.
    /// The qubit list contains all qubits that will be used in the circuit.
    BeforeCircuit { num_qubits: usize },

    /// Emitted at the end of circuit execution.
    ///
    /// Used for final state effects or cleanup.
    AfterCircuit { num_qubits: usize },

    /// Emitted between circuit layers/ticks.
    ///
    /// Used for layer-based idle noise when explicit idle durations
    /// are not tracked. The layer index indicates the layer that just completed.
    BetweenLayers {
        qubits: &'a [QubitId],
        layer_index: usize,
    },

    /// A typed signal from the command stream.
    ///
    /// Signals carry user-defined metadata that flows alongside gate commands.
    /// Use [`NoiseEvent::signal()`] for typed access and [`NoiseEvent::is_signal()`]
    /// to check the type.
    ///
    /// The `type_id` enables O(1) filtering in `responds_to()` without
    /// downcasting. The `data` field carries the actual signal value,
    /// accessible via `signal::<T>()`.
    Signal {
        /// The concrete signal type's `TypeId`, for fast filtering.
        type_id: TypeId,
        /// Type-erased signal data. Use `signal::<T>()` for typed access.
        data: &'a dyn Any,
    },
}

impl<'a> NoiseEvent<'a> {
    /// Inherited noise classification of a gate operation. The scheduled type
    /// and custom gate ID remain intact for explicit calibration and provenance.
    #[must_use]
    pub fn noise_gate_type(&self) -> Option<GateType> {
        let (gate_type, angles) = match self {
            Self::BeforeGate {
                gate_type, angles, ..
            }
            | Self::AfterGate {
                gate_type, angles, ..
            } => (*gate_type, *angles),
            _ => return None,
        };
        let gate = pecos_core::Gate::new(gate_type.into(), angles.to_vec(), Vec::new(), Vec::new());
        Some(if gate.phase_angle().is_some() {
            GateType::RZ
        } else {
            gate_type
        })
    }

    /// Get the payload-independent event kind.
    #[must_use]
    pub const fn kind(&self) -> NoiseEventKind {
        match self {
            Self::BeforeGate { .. } => NoiseEventKind::BeforeGate,
            Self::AfterGate { .. } => NoiseEventKind::AfterGate,
            Self::BeforeMeasurement { .. } => NoiseEventKind::BeforeMeasurement,
            Self::AfterMeasurement { .. } => NoiseEventKind::AfterMeasurement,
            Self::AfterPreparation { .. } => NoiseEventKind::AfterPreparation,
            Self::IdleTime { .. } => NoiseEventKind::IdleTime,
            Self::AfterReset { .. } => NoiseEventKind::AfterReset,
            Self::BeforeCircuit { .. } => NoiseEventKind::BeforeCircuit,
            Self::AfterCircuit { .. } => NoiseEventKind::AfterCircuit,
            Self::BetweenLayers { .. } => NoiseEventKind::BetweenLayers,
            Self::Signal { .. } => NoiseEventKind::Signal,
        }
    }

    /// Create a `BeforeGate` event with gate ID derived from gate type.
    ///
    /// The `gate_id` is automatically set to `gate_type.to_gate_id()`, enabling
    /// uniform gate identification in noise channels regardless of whether
    /// the gate is a core gate or custom gate.
    #[must_use]
    pub fn before_gate(gate_type: GateType, qubits: &'a [QubitId], angles: &'a [Angle64]) -> Self {
        Self::BeforeGate {
            gate_type,
            qubits,
            angles,
            gate_id: Some(gate_type.to_gate_id()),
        }
    }

    /// Create a `BeforeGate` event with an explicit custom gate ID.
    ///
    /// Use this when you need to override the gate ID (e.g., for tracking
    /// the original custom gate through decomposition).
    #[must_use]
    pub fn before_gate_with_id(
        gate_type: GateType,
        qubits: &'a [QubitId],
        angles: &'a [Angle64],
        gate_id: GateId,
    ) -> Self {
        Self::BeforeGate {
            gate_type,
            qubits,
            angles,
            gate_id: Some(gate_id),
        }
    }

    /// Create an `AfterGate` event with gate ID derived from gate type.
    ///
    /// The `gate_id` is automatically set to `gate_type.to_gate_id()`, enabling
    /// uniform gate identification in noise channels regardless of whether
    /// the gate is a core gate or custom gate.
    #[must_use]
    pub fn after_gate(gate_type: GateType, qubits: &'a [QubitId], angles: &'a [Angle64]) -> Self {
        Self::AfterGate {
            gate_type,
            qubits,
            angles,
            gate_id: Some(gate_type.to_gate_id()),
        }
    }

    /// Create an `AfterGate` event with an explicit custom gate ID.
    ///
    /// Use this when you need to override the gate ID (e.g., for tracking
    /// the original custom gate through decomposition).
    #[must_use]
    pub fn after_gate_with_id(
        gate_type: GateType,
        qubits: &'a [QubitId],
        angles: &'a [Angle64],
        gate_id: GateId,
    ) -> Self {
        Self::AfterGate {
            gate_type,
            qubits,
            angles,
            gate_id: Some(gate_id),
        }
    }

    /// Get the gate ID for gate events.
    ///
    /// Returns `Some(gate_id)` for `BeforeGate` and `AfterGate` events,
    /// `None` for other event types. For gate events created with the
    /// standard constructors, this is always populated (derived from
    /// `gate_type.to_gate_id()` or explicitly provided).
    #[must_use]
    pub fn gate_id(&self) -> Option<GateId> {
        match self {
            Self::BeforeGate { gate_id, .. } | Self::AfterGate { gate_id, .. } => *gate_id,
            _ => None,
        }
    }

    /// Get the qubits involved in this event (if qubit-specific).
    ///
    /// Returns an empty slice for circuit-level events like `BeforeCircuit`
    /// and `AfterCircuit`.
    #[must_use]
    pub fn qubits(&self) -> &[QubitId] {
        match self {
            Self::BeforeGate { qubits, .. }
            | Self::AfterGate { qubits, .. }
            | Self::BeforeMeasurement { qubits }
            | Self::AfterMeasurement { qubits, .. }
            | Self::AfterPreparation { qubits }
            | Self::IdleTime { qubits, .. }
            | Self::AfterReset { qubits }
            | Self::BetweenLayers { qubits, .. } => qubits,
            Self::BeforeCircuit { .. } | Self::AfterCircuit { .. } | Self::Signal { .. } => &[],
        }
    }

    /// Get the angles for this event (if applicable).
    #[must_use]
    pub fn angles(&self) -> &[Angle64] {
        match self {
            Self::BeforeGate { angles, .. } | Self::AfterGate { angles, .. } => angles,
            _ => &[],
        }
    }

    /// Check if this is a gate event.
    #[must_use]
    pub fn is_gate_event(&self) -> bool {
        matches!(self, Self::BeforeGate { .. } | Self::AfterGate { .. })
    }

    /// Check if this is a circuit-level event.
    #[must_use]
    pub fn is_circuit_event(&self) -> bool {
        matches!(self, Self::BeforeCircuit { .. } | Self::AfterCircuit { .. })
    }

    /// Check if this is a reset event.
    #[must_use]
    pub fn is_reset_event(&self) -> bool {
        matches!(self, Self::AfterReset { .. })
    }

    /// Check if this is a signal event.
    #[must_use]
    pub fn is_signal_event(&self) -> bool {
        matches!(self, Self::Signal { .. })
    }

    /// Create a `Signal` event from a typed signal value.
    #[must_use]
    pub fn from_signal<S: Signal>(signal: &'a S) -> Self {
        Self::Signal {
            type_id: TypeId::of::<S>(),
            data: signal,
        }
    }

    /// Try to extract a typed signal from this event.
    ///
    /// Returns `Some(&S)` if this is a `Signal` event carrying data of type `S`,
    /// `None` otherwise.
    ///
    /// ```
    /// use pecos_core::impl_signal;
    /// use pecos_neo::noise::NoiseEvent;
    ///
    /// #[derive(Copy, Clone, Debug)]
    /// struct Temperature(pub f64);
    /// impl_signal!(Temperature);
    ///
    /// let temp = Temperature(300.0);
    /// let event = NoiseEvent::from_signal(&temp);
    ///
    /// assert_eq!(event.signal::<Temperature>().unwrap().0, 300.0);
    /// ```
    #[must_use]
    pub fn signal<S: Signal>(&self) -> Option<&S> {
        match self {
            Self::Signal { data, .. } => data.downcast_ref::<S>(),
            _ => None,
        }
    }

    /// Check if this is a signal of a specific type.
    ///
    /// This is cheaper than `signal::<T>()` when you only need to check the type
    /// (no downcast needed, just `TypeId` comparison).
    ///
    /// ```
    /// use pecos_core::impl_signal;
    /// use pecos_neo::noise::NoiseEvent;
    ///
    /// #[derive(Copy, Clone, Debug)]
    /// struct Temperature(pub f64);
    /// impl_signal!(Temperature);
    ///
    /// #[derive(Copy, Clone, Debug)]
    /// struct RoundBoundary(pub i64);
    /// impl_signal!(RoundBoundary);
    ///
    /// let temp = Temperature(300.0);
    /// let event = NoiseEvent::from_signal(&temp);
    ///
    /// assert!(event.is_signal::<Temperature>());
    /// assert!(!event.is_signal::<RoundBoundary>());
    /// ```
    #[must_use]
    pub fn is_signal<S: Signal>(&self) -> bool {
        matches!(self, Self::Signal { type_id, .. } if *type_id == TypeId::of::<S>())
    }

    /// Apply inherent state updates for this event to the context.
    ///
    /// This captures the fundamental semantics of each event type:
    /// - Preparation: marks qubits as active (and clears leakage)
    /// - Measurement: marks qubits as inactive
    /// - Reset: marks qubits as prepared (clears leakage)
    ///
    /// The composer does not call this itself. `CorePlugin`'s event handlers
    /// call it for `AfterPreparation` and `AfterMeasurement` events, so this
    /// state is tracked independently of the channels added only when
    /// `CorePlugin` is registered. `NoiseModelBuilder` and
    /// `GeneralNoiseModelBuilder` register it.
    ///
    /// Note: This is separate from noise effects (which channels handle).
    /// Events have inherent meaning independent of noise.
    pub fn apply_state_updates(&self, ctx: &mut NoiseContext) {
        match self {
            Self::AfterPreparation { qubits } | Self::AfterReset { qubits } => {
                for &qubit in *qubits {
                    ctx.mark_prepared(qubit);
                }
            }
            Self::AfterMeasurement { qubits, .. } => {
                for &qubit in *qubits {
                    ctx.mark_measured(qubit);
                }
            }
            // Circuit-level, signal, and other events don't have inherent per-qubit state effects
            Self::BeforeGate { .. }
            | Self::AfterGate { .. }
            | Self::BeforeMeasurement { .. }
            | Self::IdleTime { .. }
            | Self::BeforeCircuit { .. }
            | Self::AfterCircuit { .. }
            | Self::BetweenLayers { .. }
            | Self::Signal { .. } => {}
        }
    }
}

/// Response from a noise channel indicating what errors to inject.
///
/// This enum is optimized for size by boxing the `InjectGates` variant,
/// which would otherwise dominate the enum size (296 bytes for inline storage
/// vs 8 bytes for the Box pointer). This reduces `NoiseResponse` from 304 bytes
/// to ~48 bytes, improving cache efficiency in the hot noise emission path.
#[derive(Debug, Clone, Default)]
pub enum NoiseResponse {
    /// No noise to inject.
    #[default]
    None,

    /// Inject additional gate operations after the current operation.
    ///
    /// The `SmallVec` is boxed to reduce enum size. Most noise responses
    /// inject 0-2 gates, so the heap allocation cost is acceptable.
    InjectGates(Box<SmallVec<[GateCommand; 4]>>),

    /// Suppress each operand group of the current gate containing a listed qubit.
    ///
    /// Qubits outside the current gate's operands have no effect. To suppress
    /// the whole command, list all of its operands.
    ///
    /// A suppressed group that contains no leaked qubit at the skip decision
    /// is treated as a gate that did not run: it receives no after-gate effects at all, including
    /// gate error, idle-after-two-qubit-gate noise, and acting as a crosstalk
    /// source. Its qubits can still be crosstalk targets of the groups that
    /// ran. A channel that skips a group and wants other noise on it must
    /// apply that noise itself.
    SkipGate(SmallVec<[QubitId; 4]>),

    /// Flip measurement outcomes for specific qubits.
    FlipOutcomes(SmallVec<[QubitId; 4]>),

    /// Mark qubits as leaked (left computational subspace).
    MarkLeaked(SmallVec<[QubitId; 4]>),

    /// Mark qubits as returned from leakage (seepage).
    MarkUnleaked(SmallVec<[QubitId; 4]>),

    /// Mark measurement outcomes as coming from leaked qubits.
    ///
    /// For these qubits, the measurement outcome should be reported as 2
    /// (or another special indicator) rather than 0 or 1. This matches
    /// `MeasureLeaked` behavior in `GeneralNoiseModel`.
    LeakedMeasurement(SmallVec<[QubitId; 4]>),

    /// Force measurement outcomes to specific values.
    ///
    /// Each tuple is (qubit, `forced_value`). The outcome for the qubit
    /// will be set to the forced value regardless of the actual measurement.
    ForceOutcomes(SmallVec<[(QubitId, bool); 4]>),

    /// Multiple responses to apply.
    Multiple(Vec<NoiseResponse>),
}

/// A gate-execution capability required by a configured noise mechanism.
///
/// Noise channels use this metadata to let runners reject incompatible
/// configurations before the first shot. The runtime gate dispatcher remains
/// a defensive backstop for custom channels that do not declare a requirement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoiseGateRequirement {
    /// Gate type the noise mechanism can inject.
    pub gate_type: GateType,
    /// Setter or constructor that enabled the mechanism.
    pub configured_by: &'static str,
    /// Concrete configuration change that makes the pairing valid.
    pub fix: &'static str,
}

impl NoiseGateRequirement {
    /// Create a gate-execution requirement for a noise mechanism.
    #[must_use]
    pub const fn new(gate_type: GateType, configured_by: &'static str, fix: &'static str) -> Self {
        Self {
            gate_type,
            configured_by,
            fix,
        }
    }
}

impl NoiseResponse {
    /// Create a response that injects a single gate.
    #[must_use]
    pub fn inject_gate(gate: GateCommand) -> Self {
        Self::InjectGates(Box::new(smallvec::smallvec![gate]))
    }

    /// Create a response that injects multiple gates.
    #[must_use]
    pub fn inject_gates(gates: SmallVec<[GateCommand; 4]>) -> Self {
        Self::InjectGates(Box::new(gates))
    }

    /// Combine two responses into one.
    #[must_use]
    pub fn combine(self, other: Self) -> Self {
        match (self, other) {
            (Self::None, other) => other,
            (this, Self::None) => this,
            (Self::Multiple(mut vec), Self::Multiple(other_vec)) => {
                vec.extend(other_vec);
                Self::Multiple(vec)
            }
            (Self::Multiple(mut vec), other) => {
                vec.push(other);
                Self::Multiple(vec)
            }
            (this, Self::Multiple(mut vec)) => {
                vec.insert(0, this);
                Self::Multiple(vec)
            }
            (this, other) => Self::Multiple(vec![this, other]),
        }
    }

    /// Check if this response has any effect.
    #[must_use]
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Get the union of qubits whose operand groups should be skipped.
    #[must_use]
    pub fn skipped_qubits(&self) -> SmallVec<[QubitId; 4]> {
        let mut qubits = match self {
            Self::SkipGate(qubits) => qubits.clone(),
            Self::Multiple(responses) => responses.iter().flat_map(Self::skipped_qubits).collect(),
            _ => SmallVec::new(),
        };
        qubits.sort_unstable();
        qubits.dedup();
        qubits
    }

    /// Get qubits that have leaked measurements (should return 2).
    #[must_use]
    pub fn leaked_measurements(&self) -> SmallVec<[QubitId; 4]> {
        match self {
            Self::LeakedMeasurement(qubits) => qubits.clone(),
            Self::Multiple(responses) => responses
                .iter()
                .flat_map(Self::leaked_measurements)
                .collect(),
            _ => SmallVec::new(),
        }
    }
}

/// Retain complete operand groups that contain no skipped qubits, in order.
/// The group size must be positive. Unchanged operands are borrowed.
pub(crate) fn surviving_gate_qubits<'a>(
    qubits: &'a [QubitId],
    group_size: usize,
    skipped: &[QubitId],
) -> Cow<'a, [QubitId]> {
    debug_assert!(
        group_size > 0,
        "callers must pass a positive operand-group size"
    );
    if skipped.is_empty() || !qubits.iter().any(|q| skipped.contains(q)) {
        return Cow::Borrowed(qubits);
    }
    Cow::Owned(
        qubits
            .chunks(group_size)
            .filter(|group| !group.iter().any(|q| skipped.contains(q)))
            .flatten()
            .copied()
            .collect(),
    )
}

/// Partition operand groups for execution and after-gate dispatch, in order.
/// Suppressed groups are retained only for after-gate dispatch when they contain
/// a leaked qubit at this decision point. The group size must be positive.
/// When no operands are skipped, both slices are borrowed.
pub(crate) fn partition_gate_qubits<'a>(
    qubits: &'a [QubitId],
    group_size: usize,
    skipped: &[QubitId],
    mut is_leaked: impl FnMut(QubitId) -> bool,
) -> (Cow<'a, [QubitId]>, Cow<'a, [QubitId]>) {
    debug_assert!(
        group_size > 0,
        "callers must pass a positive operand-group size"
    );
    if skipped.is_empty() || !qubits.iter().any(|q| skipped.contains(q)) {
        return (Cow::Borrowed(qubits), Cow::Borrowed(qubits));
    }
    let mut executed = Vec::new();
    let mut after_gate = Vec::new();
    for group in qubits.chunks(group_size) {
        if !group.iter().any(|q| skipped.contains(q)) {
            executed.extend_from_slice(group);
            after_gate.extend_from_slice(group);
        } else if group.iter().copied().any(&mut is_leaked) {
            after_gate.extend_from_slice(group);
        }
    }
    (Cow::Owned(executed), Cow::Owned(after_gate))
}

/// Trait for noise channels that respond to events.
///
/// Each noise channel handles a specific type of noise (e.g., single-qubit depolarizing,
/// measurement errors, leakage). The composer combines multiple channels to form
/// a complete noise model.
pub trait NoiseChannel: Send + Sync {
    /// Declare every kind on which `try_apply` can return a response.
    ///
    /// This must depend only on construction-time configuration and must not
    /// change after insertion into a model. The model caches the declaration.
    /// Extra kinds are safe; missing a supported kind silently drops noise.
    fn event_kinds(&self) -> EventKinds {
        EventKinds::ALL
    }

    /// Check if this channel responds to a given event.
    ///
    /// This is an optimization to avoid calling `apply` for events the channel
    /// doesn't handle.
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool;

    /// Apply noise in response to an event.
    ///
    /// # Arguments
    /// * `event` - The event that triggered this call
    /// * `ctx` - Shared noise context (leakage state, etc.)
    /// * `rng` - Random number generator for stochastic noise
    ///
    /// # Returns
    /// A response indicating what noise to inject, if any.
    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> NoiseResponse;

    /// Combined `responds_to` + `apply` in one call for better performance.
    ///
    /// Returns `None` if this channel doesn't respond to the event,
    /// otherwise returns the noise response (which may be `NoiseResponse::None`).
    ///
    /// The default implementation calls `responds_to` then `apply`, but channels
    /// can override this for better performance by avoiding redundant checks.
    #[inline]
    fn try_apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> Option<NoiseResponse> {
        if self.responds_to(event) {
            Some(self.apply(event, ctx, rng))
        } else {
            None
        }
    }

    /// Get a human-readable name for this channel.
    fn name(&self) -> &'static str;

    /// Get the priority of this channel (higher = earlier).
    ///
    /// Channels with higher priority are applied first. This is useful for
    /// ensuring certain effects (like leakage checks) happen before others.
    fn priority(&self) -> i32 {
        0
    }

    /// Report injected gates whose runner support must be validated.
    ///
    /// The default is empty for channels that emit no gates or only gates every
    /// target runner supports. Custom channels that can emit rotations or other
    /// runner-dependent gates should override this so mismatches fail during
    /// configuration; runtime dispatch will panic if an undeclared unsupported
    /// gate is reached.
    fn gate_requirements(&self) -> SmallVec<[NoiseGateRequirement; 2]> {
        SmallVec::new()
    }

    /// Clone this channel into a boxed trait object.
    ///
    /// Required for cloning `ComposableNoiseModel` to support parallel execution.
    fn clone_box(&self) -> Box<dyn NoiseChannel>;
}

/// Distribution of Pauli errors for single-qubit noise.
///
/// Weights should sum to 1.0. For uniform depolarizing, use equal weights.
#[derive(Debug, Clone, Copy)]
pub struct PauliWeights {
    /// Weight for X errors.
    pub(crate) x: f64,
    /// Weight for Y errors.
    pub(crate) y: f64,
    /// Weight for Z errors.
    pub(crate) z: f64,
}

impl Default for PauliWeights {
    fn default() -> Self {
        Self::uniform()
    }
}

impl PauliWeights {
    /// Return the validated `x` weight.
    #[must_use]
    pub fn x(&self) -> f64 {
        self.x
    }

    /// Return the validated `y` weight.
    #[must_use]
    pub fn y(&self) -> f64 {
        self.y
    }

    /// Return the validated `z` weight.
    #[must_use]
    pub fn z(&self) -> f64 {
        self.z
    }

    /// Distribution containing only Z errors.
    #[must_use]
    pub fn z_only() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        }
    }

    /// Distribution containing only X errors.
    #[must_use]
    pub fn x_only() -> Self {
        Self {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Validate this distribution before use by a noise channel.
    ///
    /// # Errors
    /// Rejects invalid weights or a total other than one.
    pub fn validate(&self) -> ValidationResult<()> {
        validate_distribution(&[self.x, self.y, self.z], "PauliWeights")?;
        Ok(())
    }

    /// Uniform distribution (1/3 each).
    #[must_use]
    pub fn uniform() -> Self {
        Self {
            x: 1.0 / 3.0,
            y: 1.0 / 3.0,
            z: 1.0 / 3.0,
        }
    }

    /// Z-biased distribution (mostly dephasing).
    ///
    /// # Errors
    /// Returns an error for invalid probabilities or weights.
    pub fn z_biased(z_weight: f64) -> ValidationResult<Self> {
        validate_probability(z_weight, "z_weight")?;
        let remaining = 1.0 - z_weight;
        Ok(Self {
            x: remaining / 2.0,
            y: remaining / 2.0,
            z: z_weight,
        })
    }

    /// X-biased distribution (mostly bit-flip).
    ///
    /// # Errors
    /// Returns an error for invalid probabilities or weights.
    pub fn x_biased(x_weight: f64) -> ValidationResult<Self> {
        validate_probability(x_weight, "x_weight")?;
        let remaining = 1.0 - x_weight;
        Ok(Self {
            x: x_weight,
            y: remaining / 2.0,
            z: remaining / 2.0,
        })
    }

    /// Custom distribution.
    ///
    /// # Errors
    /// Rejects invalid weights or a total other than one.
    pub fn custom(x: f64, y: f64, z: f64) -> ValidationResult<Self> {
        validate_distribution(&[x, y, z], "PauliWeights")?;
        Ok(Self { x, y, z })
    }

    /// Sample a Pauli gate type based on the weights.
    #[must_use]
    pub fn sample(&self, r: f64) -> GateType {
        if r < self.x {
            GateType::X
        } else if r < self.x + self.y {
            GateType::Y
        } else {
            GateType::Z
        }
    }
}

/// The 15 non-identity two-qubit Pauli operators as (first, second) pairs.
pub const TWO_QUBIT_PAULIS: [(GateType, GateType); 15] = [
    (GateType::X, GateType::I),
    (GateType::Y, GateType::I),
    (GateType::Z, GateType::I),
    (GateType::I, GateType::X),
    (GateType::I, GateType::Y),
    (GateType::I, GateType::Z),
    (GateType::X, GateType::X),
    (GateType::X, GateType::Y),
    (GateType::X, GateType::Z),
    (GateType::Y, GateType::X),
    (GateType::Y, GateType::Y),
    (GateType::Y, GateType::Z),
    (GateType::Z, GateType::X),
    (GateType::Z, GateType::Y),
    (GateType::Z, GateType::Z),
];

/// Distribution of Pauli errors for two-qubit noise.
///
/// Represents weights for the 15 non-identity two-qubit Pauli operators.
/// For uniform depolarizing, each has weight 1/15.
///
/// The operators are ordered as: XI, YI, ZI, IX, IY, IZ, XX, XY, XZ, YX, YY, YZ, ZX, ZY, ZZ
#[derive(Debug, Clone)]
pub struct TwoQubitPauliWeights {
    /// Weights for each of the 15 two-qubit Pauli operators.
    pub(crate) weights: [f64; 15],
}

impl Default for TwoQubitPauliWeights {
    fn default() -> Self {
        Self::uniform()
    }
}

impl TwoQubitPauliWeights {
    /// Return the validated `weights` weight.
    #[must_use]
    pub fn weights(&self) -> [f64; 15] {
        self.weights
    }

    /// Validate this distribution before use by a noise channel.
    ///
    /// # Errors
    /// Rejects invalid weights or a total other than one.
    pub fn validate(&self) -> ValidationResult<()> {
        validate_distribution(&self.weights, "TwoQubitPauliWeights")?;
        Ok(())
    }

    /// Uniform distribution (1/15 each).
    #[must_use]
    pub fn uniform() -> Self {
        Self {
            weights: [1.0 / 15.0; 15],
        }
    }

    /// Create from custom weights.
    ///
    /// # Errors
    /// Rejects invalid weights or a total other than one.
    pub fn custom(weights: [f64; 15]) -> ValidationResult<Self> {
        validate_distribution(&weights, "TwoQubitPauliWeights")?;
        Ok(Self { weights })
    }

    /// Create Z-biased weights (more dephasing on both qubits).
    ///
    /// Weight for ZZ is `zz_weight`, remaining weight is distributed uniformly.
    ///
    /// # Errors
    /// Returns an error for invalid probabilities or weights.
    pub fn zz_biased(zz_weight: f64) -> ValidationResult<Self> {
        validate_probability(zz_weight, "zz_weight")?;
        let remaining = 1.0 - zz_weight;
        let other = remaining / 14.0;
        let mut weights = [other; 15];
        weights[14] = zz_weight; // ZZ is at index 14
        Ok(Self { weights })
    }

    /// Sample a two-qubit Pauli operator based on the weights.
    ///
    /// Returns the index into `TWO_QUBIT_PAULIS`.
    #[must_use]
    pub fn sample(&self, r: f64) -> usize {
        let mut cumulative = 0.0;
        for (i, &w) in self.weights.iter().enumerate() {
            cumulative += w;
            if r < cumulative {
                return i;
            }
        }
        // Fallback to last (shouldn't happen with normalized weights)
        14
    }

    /// Get the Pauli operators for a sampled index.
    #[must_use]
    pub fn get_paulis(index: usize) -> (GateType, GateType) {
        TWO_QUBIT_PAULIS[index.min(14)]
    }
}

// ============================================================================
// Emission Weights (include leakage as an option)
// ============================================================================

/// Result of sampling from an emission distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SingleQubitEmissionResult {
    /// Apply a Pauli gate.
    Pauli(GateType),
    /// Qubit leaked out of computational subspace.
    Leaked,
}

/// Distribution of emission errors for single-qubit gates.
///
/// Emission errors can cause either Pauli errors OR leakage.
/// Weights should sum to 1.0.
///
/// This matches `GeneralNoiseModel`'s `p1_emission_model` which can
/// include "X", "Y", "Z", or "L" (leakage).
#[derive(Debug, Clone, Copy)]
pub struct SingleQubitEmissionWeights {
    /// Weight for X errors.
    pub(crate) x: f64,
    /// Weight for Y errors.
    pub(crate) y: f64,
    /// Weight for Z errors.
    pub(crate) z: f64,
    /// Weight for leakage.
    pub(crate) leak: f64,
}

impl Default for SingleQubitEmissionWeights {
    fn default() -> Self {
        Self::uniform()
    }
}

impl SingleQubitEmissionWeights {
    /// Return the validated `x` weight.
    #[must_use]
    pub fn x(&self) -> f64 {
        self.x
    }

    /// Return the validated `y` weight.
    #[must_use]
    pub fn y(&self) -> f64 {
        self.y
    }

    /// Return the validated `z` weight.
    #[must_use]
    pub fn z(&self) -> f64 {
        self.z
    }

    /// Return the validated `leak` weight.
    #[must_use]
    pub fn leak(&self) -> f64 {
        self.leak
    }

    /// Validate this distribution before use by a noise channel.
    ///
    /// # Errors
    /// Rejects invalid weights or a total other than one.
    pub fn validate(&self) -> ValidationResult<()> {
        validate_distribution(
            &[self.x, self.y, self.z, self.leak],
            "SingleQubitEmissionWeights",
        )?;
        Ok(())
    }

    /// Uniform distribution over Pauli errors (no leakage).
    #[must_use]
    pub fn uniform() -> Self {
        Self {
            x: 1.0 / 3.0,
            y: 1.0 / 3.0,
            z: 1.0 / 3.0,
            leak: 0.0,
        }
    }

    /// Uniform distribution including leakage (1/4 each).
    #[must_use]
    pub fn uniform_with_leakage() -> Self {
        Self {
            x: 0.25,
            y: 0.25,
            z: 0.25,
            leak: 0.25,
        }
    }

    /// Leakage-only emission (no Pauli errors).
    #[must_use]
    pub fn leakage_only() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            leak: 1.0,
        }
    }

    /// Custom distribution.
    ///
    /// # Errors
    /// Rejects invalid weights or a total other than one.
    pub fn custom(x: f64, y: f64, z: f64, leak: f64) -> ValidationResult<Self> {
        validate_distribution(&[x, y, z, leak], "SingleQubitEmissionWeights")?;
        Ok(Self { x, y, z, leak })
    }

    /// Sample an emission result based on the weights.
    #[must_use]
    pub fn sample(&self, r: f64) -> SingleQubitEmissionResult {
        if r < self.x {
            SingleQubitEmissionResult::Pauli(GateType::X)
        } else if r < self.x + self.y {
            SingleQubitEmissionResult::Pauli(GateType::Y)
        } else if r < self.x + self.y + self.z {
            SingleQubitEmissionResult::Pauli(GateType::Z)
        } else {
            SingleQubitEmissionResult::Leaked
        }
    }
}

/// Result of sampling from a two-qubit emission distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwoQubitEmissionResult {
    /// Pauli to apply to first qubit (None if leaked or identity).
    pub first: Option<GateType>,
    /// Pauli to apply to second qubit (None if leaked or identity).
    pub second: Option<GateType>,
    /// Whether first qubit leaked.
    pub first_leaked: bool,
    /// Whether second qubit leaked.
    pub second_leaked: bool,
}

impl TwoQubitEmissionResult {
    /// Check if any leakage occurred.
    #[must_use]
    pub fn has_leakage(&self) -> bool {
        self.first_leaked || self.second_leaked
    }
}

/// Two-qubit emission operators including leakage.
///
/// Format: (first, second) where each is X, Y, Z, I (identity), or L (leakage).
/// "II" is not included as it represents no operation.
/// Total of 24 operators: 5x5 - 1 = 24
pub const TWO_QUBIT_EMISSION_OPS: [(char, char); 24] = [
    // First qubit X
    ('X', 'I'),
    ('X', 'X'),
    ('X', 'Y'),
    ('X', 'Z'),
    ('X', 'L'),
    // First qubit Y
    ('Y', 'I'),
    ('Y', 'X'),
    ('Y', 'Y'),
    ('Y', 'Z'),
    ('Y', 'L'),
    // First qubit Z
    ('Z', 'I'),
    ('Z', 'X'),
    ('Z', 'Y'),
    ('Z', 'Z'),
    ('Z', 'L'),
    // First qubit I (identity)
    ('I', 'X'),
    ('I', 'Y'),
    ('I', 'Z'),
    ('I', 'L'),
    // First qubit L (leakage)
    ('L', 'I'),
    ('L', 'X'),
    ('L', 'Y'),
    ('L', 'Z'),
    ('L', 'L'),
];

/// Distribution of emission errors for two-qubit gates.
///
/// Emission errors can cause Pauli errors on either/both qubits,
/// and/or leakage on either/both qubits.
///
/// This matches `GeneralNoiseModel`'s `p2_emission_model` which uses
/// two-character keys like "XY", "IL", "LX", etc.
#[derive(Debug, Clone)]
pub struct TwoQubitEmissionWeights {
    /// Weights for each of the 24 two-qubit emission operators.
    pub(crate) weights: [f64; 24],
}

impl Default for TwoQubitEmissionWeights {
    fn default() -> Self {
        Self::uniform_pauli()
    }
}

impl TwoQubitEmissionWeights {
    /// Return the validated `weights` weight.
    #[must_use]
    pub fn weights(&self) -> [f64; 24] {
        self.weights
    }

    /// Validate this distribution before use by a noise channel.
    ///
    /// # Errors
    /// Rejects invalid weights or a total other than one.
    pub fn validate(&self) -> ValidationResult<()> {
        validate_distribution(&self.weights, "TwoQubitEmissionWeights")?;
        Ok(())
    }

    /// Uniform distribution over Pauli errors only (no leakage).
    ///
    /// Matches the 15 non-identity two-qubit Pauli operators.
    #[must_use]
    pub fn uniform_pauli() -> Self {
        let mut weights = [0.0; 24];
        // Set equal weights for the 15 Pauli-only entries
        let pauli_indices = [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 15, 16, 17];
        for &i in &pauli_indices {
            weights[i] = 1.0 / 15.0;
        }
        Self { weights }
    }

    /// Uniform distribution including leakage (1/24 each).
    #[must_use]
    pub fn uniform_with_leakage() -> Self {
        Self {
            weights: [1.0 / 24.0; 24],
        }
    }

    /// Create from custom weights.
    ///
    /// # Errors
    /// Rejects invalid weights or a total other than one.
    pub fn custom(weights: [f64; 24]) -> ValidationResult<Self> {
        validate_distribution(&weights, "TwoQubitEmissionWeights")?;
        Ok(Self { weights })
    }

    /// Sample a two-qubit emission operator based on the weights.
    ///
    /// Returns the index into `TWO_QUBIT_EMISSION_OPS`.
    #[must_use]
    pub fn sample(&self, r: f64) -> usize {
        let mut cumulative = 0.0;
        for (i, &w) in self.weights.iter().enumerate() {
            cumulative += w;
            if r < cumulative {
                return i;
            }
        }
        // Fallback to last
        23
    }

    /// Get the emission result for a sampled index.
    #[must_use]
    pub fn get_result(index: usize) -> TwoQubitEmissionResult {
        let (first_char, second_char) = TWO_QUBIT_EMISSION_OPS[index.min(23)];

        let (first, first_leaked) = match first_char {
            'X' => (Some(GateType::X), false),
            'Y' => (Some(GateType::Y), false),
            'Z' => (Some(GateType::Z), false),
            'I' => (None, false),
            'L' => (None, true),
            _ => unreachable!(),
        };

        let (second, second_leaked) = match second_char {
            'X' => (Some(GateType::X), false),
            'Y' => (Some(GateType::Y), false),
            'Z' => (Some(GateType::Z), false),
            'I' => (None, false),
            'L' => (None, true),
            _ => unreachable!(),
        };

        TwoQubitEmissionResult {
            first,
            second,
            first_leaked,
            second_leaked,
        }
    }
}

// ============================================================================
// Crosstalk Transition Weights
// ============================================================================

/// Result of sampling from a crosstalk distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrosstalkResult {
    /// No change (stay at current value).
    NoChange,
    /// Flip the bit.
    Flip,
    /// Transition to leaked state.
    Leak,
}

/// Transition probabilities for crosstalk effects on a single qubit.
///
/// Models what happens to a qubit when it experiences crosstalk from
/// a nearby measurement. The transitions depend on the qubit's current state.
///
/// This matches `GeneralNoiseModel`'s `p_meas_crosstalk_model` which uses
/// keys like "0->0", "0->1", "0->L", "1->0", "1->1", "1->L".
#[derive(Debug, Clone, Copy)]
pub struct CrosstalkTransitions {
    /// Probability of staying at 0 when starting at 0.
    pub(crate) zero_stay: f64,
    /// Probability of flipping to 1 when starting at 0.
    pub(crate) zero_flip: f64,
    /// Probability of leaking when starting at 0.
    pub(crate) zero_leak: f64,
    /// Probability of staying at 1 when starting at 1.
    pub(crate) one_stay: f64,
    /// Probability of flipping to 0 when starting at 1.
    pub(crate) one_flip: f64,
    /// Probability of leaking when starting at 1.
    pub(crate) one_leak: f64,
}

impl Default for CrosstalkTransitions {
    fn default() -> Self {
        Self::flip_only()
    }
}

impl CrosstalkTransitions {
    /// Return the validated `from_0_stay` weight.
    #[must_use]
    pub fn from_0_stay(&self) -> f64 {
        self.zero_stay
    }

    /// Return the validated `from_0_flip` weight.
    #[must_use]
    pub fn from_0_flip(&self) -> f64 {
        self.zero_flip
    }

    /// Return the validated `from_0_leak` weight.
    #[must_use]
    pub fn from_0_leak(&self) -> f64 {
        self.zero_leak
    }

    /// Return the validated `from_1_stay` weight.
    #[must_use]
    pub fn from_1_stay(&self) -> f64 {
        self.one_stay
    }

    /// Return the validated `from_1_flip` weight.
    #[must_use]
    pub fn from_1_flip(&self) -> f64 {
        self.one_flip
    }

    /// Return the validated `from_1_leak` weight.
    #[must_use]
    pub fn from_1_leak(&self) -> f64 {
        self.one_leak
    }

    /// Validate this distribution before use by a noise channel.
    ///
    /// # Errors
    /// Rejects invalid weights or a total other than one.
    pub fn validate(&self) -> ValidationResult<()> {
        validate_distribution(
            &[self.zero_stay, self.zero_flip, self.zero_leak],
            "CrosstalkTransitions from 0",
        )?;
        validate_distribution(
            &[self.one_stay, self.one_flip, self.one_leak],
            "CrosstalkTransitions from 1",
        )?;
        Ok(())
    }

    /// Simple flip model: 50% chance to flip, no leakage.
    #[must_use]
    pub fn flip_only() -> Self {
        Self {
            zero_stay: 0.5,
            zero_flip: 0.5,
            zero_leak: 0.0,
            one_stay: 0.5,
            one_flip: 0.5,
            one_leak: 0.0,
        }
    }

    /// Symmetric model with leakage.
    ///
    /// Equal probability of stay, flip, or leak for both starting states.
    #[must_use]
    pub fn symmetric_with_leakage() -> Self {
        Self {
            zero_stay: 1.0 / 3.0,
            zero_flip: 1.0 / 3.0,
            zero_leak: 1.0 / 3.0,
            one_stay: 1.0 / 3.0,
            one_flip: 1.0 / 3.0,
            one_leak: 1.0 / 3.0,
        }
    }

    /// Custom transitions.
    ///
    /// # Errors
    /// Rejects invalid probabilities or a total other than one for either state.
    pub fn custom(
        from_0_stay: f64,
        from_0_flip: f64,
        from_0_leak: f64,
        from_1_stay: f64,
        from_1_flip: f64,
        from_1_leak: f64,
    ) -> ValidationResult<Self> {
        validate_distribution(
            &[from_0_stay, from_0_flip, from_0_leak],
            "CrosstalkTransitions from 0",
        )?;
        validate_distribution(
            &[from_1_stay, from_1_flip, from_1_leak],
            "CrosstalkTransitions from 1",
        )?;
        Ok(Self {
            zero_stay: from_0_stay,
            zero_flip: from_0_flip,
            zero_leak: from_0_leak,
            one_stay: from_1_stay,
            one_flip: from_1_flip,
            one_leak: from_1_leak,
        })
    }

    /// Sample a crosstalk result based on the qubit's current state.
    ///
    /// # Arguments
    /// * `current_state` - The qubit's current state (false = 0, true = 1)
    /// * `r` - Random value in [0, 1)
    #[must_use]
    pub fn sample(&self, current_state: bool, r: f64) -> CrosstalkResult {
        if current_state {
            // Starting from 1
            if r < self.one_stay {
                CrosstalkResult::NoChange
            } else if r < self.one_stay + self.one_flip {
                CrosstalkResult::Flip
            } else {
                CrosstalkResult::Leak
            }
        } else {
            // Starting from 0
            if r < self.zero_stay {
                CrosstalkResult::NoChange
            } else if r < self.zero_stay + self.zero_flip {
                CrosstalkResult::Flip
            } else {
                CrosstalkResult::Leak
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_event_qubits() {
        let qubits = [QubitId(0), QubitId(1)];
        let angles = [];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::CX,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        assert_eq!(event.qubits(), &qubits);
    }

    #[test]
    fn test_noise_event_angles() {
        let qubits = [QubitId(0)];
        let angles = [Angle64::QUARTER_TURN];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::RZ,
            qubits: &qubits,
            angles: &angles,
            gate_id: None,
        };

        assert_eq!(event.angles(), &angles);
    }

    #[test]
    fn test_noise_response_combine() {
        let r1 = NoiseResponse::inject_gate(GateCommand::x(QubitId(0)));
        let r2 = NoiseResponse::inject_gate(GateCommand::z(QubitId(1)));

        let combined = r1.combine(r2);

        assert!(matches!(combined, NoiseResponse::Multiple(_)));
    }

    #[test]
    fn test_noise_response_skip_gate() {
        let skip = NoiseResponse::SkipGate(smallvec::smallvec![QubitId(2), QubitId(2)]);
        assert_eq!(skip.skipped_qubits().as_slice(), &[QubitId(2)]);
        assert!(NoiseResponse::None.skipped_qubits().is_empty());

        let combined = NoiseResponse::Multiple(vec![
            skip,
            NoiseResponse::Multiple(vec![
                NoiseResponse::None,
                NoiseResponse::SkipGate(smallvec::smallvec![QubitId(0), QubitId(2)]),
            ]),
        ]);
        assert_eq!(
            combined.skipped_qubits().as_slice(),
            &[QubitId(0), QubitId(2)]
        );
    }

    #[test]
    fn test_surviving_gate_qubits() {
        let qubits = [
            QubitId(0),
            QubitId(1),
            QubitId(2),
            QubitId(3),
            QubitId(4),
            QubitId(5),
        ];
        for skipped in [&[][..], &[QubitId(99)][..]] {
            let result = surviving_gate_qubits(&qubits, 2, skipped);
            assert!(matches!(result, Cow::Borrowed(_)));
            assert_eq!(result.as_ptr(), qubits.as_ptr());
        }
        assert_eq!(
            surviving_gate_qubits(&qubits, 2, &[QubitId(2), QubitId(99)]).as_ref(),
            &[QubitId(0), QubitId(1), QubitId(4), QubitId(5)]
        );
        assert_eq!(
            surviving_gate_qubits(&qubits, 1, &[QubitId(2)]).as_ref(),
            &[QubitId(0), QubitId(1), QubitId(3), QubitId(4), QubitId(5)]
        );
        assert_eq!(
            surviving_gate_qubits(&qubits, 3, &[QubitId(2)]).as_ref(),
            &[QubitId(3), QubitId(4), QubitId(5)]
        );
        assert!(
            surviving_gate_qubits(&qubits, 2, &[QubitId(0), QubitId(2), QubitId(5)]).is_empty()
        );
    }

    #[test]
    fn test_partition_gate_qubits() {
        let qubits: Vec<_> = (0..6).map(QubitId).collect();
        for skipped in [&[][..], &[QubitId(99)][..]] {
            let (executed, after) = partition_gate_qubits(&qubits, 2, skipped, |_| {
                panic!("no leakage lookup needed without suppressed groups")
            });
            for result in [executed, after] {
                assert!(matches!(result, Cow::Borrowed(_)));
                assert_eq!(result.as_ptr(), qubits.as_ptr());
            }
        }
        for (group_size, skipped, leaked, expected_executed, expected_after) in [
            (2, vec![2, 99], vec![], vec![0, 1, 4, 5], vec![0, 1, 4, 5]),
            (
                2,
                vec![2],
                vec![3],
                vec![0, 1, 4, 5],
                vec![0, 1, 2, 3, 4, 5],
            ),
            (2, vec![0, 2], vec![1], vec![4, 5], vec![0, 1, 4, 5]),
            (
                2,
                vec![2],
                vec![2],
                vec![0, 1, 4, 5],
                vec![0, 1, 2, 3, 4, 5],
            ),
            (2, vec![0, 2, 5], vec![], vec![], vec![]),
            (1, vec![2], vec![], vec![0, 1, 3, 4, 5], vec![0, 1, 3, 4, 5]),
            (3, vec![2], vec![], vec![3, 4, 5], vec![3, 4, 5]),
        ] {
            let skipped: Vec<_> = skipped.into_iter().map(QubitId).collect();
            let (executed, after) =
                partition_gate_qubits(&qubits, group_size, &skipped, |q| leaked.contains(&q.0));
            assert_eq!(
                executed.as_ref(),
                expected_executed
                    .into_iter()
                    .map(QubitId)
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                after.as_ref(),
                expected_after.into_iter().map(QubitId).collect::<Vec<_>>()
            );
        }
        let (executed, after) = partition_gate_qubits(&[], 1, &[QubitId(0)], |_| true);
        for result in [executed, after] {
            assert!(matches!(result, Cow::Borrowed(_)));
            assert!(result.is_empty());
        }
    }

    #[test]
    fn test_pauli_weights_uniform() {
        let weights = PauliWeights::uniform();
        assert!((weights.x - 1.0 / 3.0).abs() < 1e-10);
        assert!((weights.y - 1.0 / 3.0).abs() < 1e-10);
        assert!((weights.z - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_weights_sampling() {
        let weights = PauliWeights::uniform();

        // X region: [0, 1/3)
        assert_eq!(weights.sample(0.0), GateType::X);
        assert_eq!(weights.sample(0.2), GateType::X);

        // Y region: [1/3, 2/3)
        assert_eq!(weights.sample(0.4), GateType::Y);
        assert_eq!(weights.sample(0.5), GateType::Y);

        // Z region: [2/3, 1)
        assert_eq!(weights.sample(0.8), GateType::Z);
        assert_eq!(weights.sample(0.99), GateType::Z);
    }

    #[test]
    fn test_two_qubit_pauli_weights_uniform() {
        let weights = TwoQubitPauliWeights::uniform();
        for w in &weights.weights {
            assert!((*w - 1.0 / 15.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_two_qubit_pauli_weights_sampling() {
        let weights = TwoQubitPauliWeights::uniform();

        // First Pauli (XI) should be sampled at r < 1/15
        assert_eq!(weights.sample(0.0), 0);
        assert_eq!(weights.sample(0.01), 0);

        // Last Pauli (ZZ) should be sampled at r >= 14/15
        assert_eq!(weights.sample(0.99), 14);
    }

    #[test]
    fn test_two_qubit_pauli_weights_zz_biased() {
        let weights = TwoQubitPauliWeights::zz_biased(0.5).unwrap();

        // ZZ (index 14) should have 50% weight
        assert!((weights.weights[14] - 0.5).abs() < 1e-10);

        // Other 14 operators should share the remaining 50%
        for i in 0..14 {
            assert!((weights.weights[i] - 0.5 / 14.0).abs() < 1e-10);
        }

        // Values >= 0.5 should sample ZZ
        assert_eq!(weights.sample(0.5), 14);
        assert_eq!(weights.sample(0.99), 14);
    }

    #[test]
    fn test_two_qubit_paulis_ordering() {
        // Verify the ordering matches the documentation
        assert_eq!(
            TwoQubitPauliWeights::get_paulis(0),
            (GateType::X, GateType::I)
        ); // XI
        assert_eq!(
            TwoQubitPauliWeights::get_paulis(3),
            (GateType::I, GateType::X)
        ); // IX
        assert_eq!(
            TwoQubitPauliWeights::get_paulis(6),
            (GateType::X, GateType::X)
        ); // XX
        assert_eq!(
            TwoQubitPauliWeights::get_paulis(14),
            (GateType::Z, GateType::Z)
        ); // ZZ
    }

    // ========================================================================
    // Emission Weights Tests
    // ========================================================================

    #[test]
    fn test_single_qubit_emission_uniform() {
        let weights = SingleQubitEmissionWeights::uniform();
        assert!((weights.x - 1.0 / 3.0).abs() < 1e-10);
        assert!((weights.y - 1.0 / 3.0).abs() < 1e-10);
        assert!((weights.z - 1.0 / 3.0).abs() < 1e-10);
        assert!((weights.leak).abs() < 1e-10);
    }

    #[test]
    fn test_single_qubit_emission_with_leakage() {
        let weights = SingleQubitEmissionWeights::uniform_with_leakage();
        assert!((weights.x - 0.25).abs() < 1e-10);
        assert!((weights.leak - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_single_qubit_emission_sampling() {
        let weights = SingleQubitEmissionWeights::uniform_with_leakage();

        // X region: [0, 0.25)
        assert_eq!(
            weights.sample(0.1),
            SingleQubitEmissionResult::Pauli(GateType::X)
        );

        // Y region: [0.25, 0.5)
        assert_eq!(
            weights.sample(0.3),
            SingleQubitEmissionResult::Pauli(GateType::Y)
        );

        // Z region: [0.5, 0.75)
        assert_eq!(
            weights.sample(0.6),
            SingleQubitEmissionResult::Pauli(GateType::Z)
        );

        // Leak region: [0.75, 1.0)
        assert_eq!(weights.sample(0.9), SingleQubitEmissionResult::Leaked);
    }

    #[test]
    fn test_single_qubit_emission_leakage_only() {
        let weights = SingleQubitEmissionWeights::leakage_only();
        // All samples should give leakage
        assert_eq!(weights.sample(0.0), SingleQubitEmissionResult::Leaked);
        assert_eq!(weights.sample(0.5), SingleQubitEmissionResult::Leaked);
        assert_eq!(weights.sample(0.99), SingleQubitEmissionResult::Leaked);
    }

    #[test]
    fn test_two_qubit_emission_result() {
        // Test XI (index 0)
        let result = TwoQubitEmissionWeights::get_result(0);
        assert_eq!(result.first, Some(GateType::X));
        assert_eq!(result.second, None);
        assert!(!result.has_leakage());

        // Test XL (index 4)
        let result = TwoQubitEmissionWeights::get_result(4);
        assert_eq!(result.first, Some(GateType::X));
        assert!(result.second_leaked);
        assert!(result.has_leakage());

        // Test LL (index 23)
        let result = TwoQubitEmissionWeights::get_result(23);
        assert!(result.first_leaked);
        assert!(result.second_leaked);
        assert!(result.has_leakage());
    }

    // ========================================================================
    // Crosstalk Transitions Tests
    // ========================================================================

    #[test]
    fn test_crosstalk_flip_only() {
        let trans = CrosstalkTransitions::flip_only();

        // Starting from 0
        assert_eq!(trans.sample(false, 0.25), CrosstalkResult::NoChange);
        assert_eq!(trans.sample(false, 0.75), CrosstalkResult::Flip);

        // Starting from 1
        assert_eq!(trans.sample(true, 0.25), CrosstalkResult::NoChange);
        assert_eq!(trans.sample(true, 0.75), CrosstalkResult::Flip);
    }

    #[test]
    fn test_crosstalk_with_leakage() {
        let trans = CrosstalkTransitions::symmetric_with_leakage();

        // Starting from 0: [0, 1/3) -> stay, [1/3, 2/3) -> flip, [2/3, 1) -> leak
        assert_eq!(trans.sample(false, 0.1), CrosstalkResult::NoChange);
        assert_eq!(trans.sample(false, 0.5), CrosstalkResult::Flip);
        assert_eq!(trans.sample(false, 0.9), CrosstalkResult::Leak);

        // Starting from 1: same distribution
        assert_eq!(trans.sample(true, 0.1), CrosstalkResult::NoChange);
        assert_eq!(trans.sample(true, 0.5), CrosstalkResult::Flip);
        assert_eq!(trans.sample(true, 0.9), CrosstalkResult::Leak);
    }
}
