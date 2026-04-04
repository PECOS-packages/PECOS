//! Extensible gate system for user-defined gates.
//!
//! This module provides:
//! - `GateId`: Compact identifier for gate types (core: 0-255, user: 256+)
//! - `GateSpec`: Metadata describing a gate's properties
//! - `GateRegistry`: Registry for gate specifications (scoped, not global)
//! - `GateCanonicalizer`: Maps parameterized gates to fixed gates at exact angles
//! - `AngleSnapper`: Tolerance-based angle snapping for floating-point input
//! - `CircuitValidator`: Trait and implementations for circuit validation
//! - `GateAdaptor`: Trait for decomposing gates into supported primitives

mod adaptor;
mod bridge;
mod canonicalizer;
mod gate_id;
mod gate_spec;
mod op_builder;
mod operation;
mod pauli;
mod queue_validation;
mod registry;
mod snapper;
mod stabilizer_adaptor;
mod support_set;
mod validator;
#[macro_use]
mod circuit_macro;
mod batch;
mod decomposition;
mod definitions;
mod noise_integration;
mod plugin;
mod user_gates;

pub use adaptor::{
    AdaptedGate, CompositeAdaptor, CompositeExtendedAdaptor, CustomAdaptor, GateAdaptor,
    LiftedAdaptor, StandardAdaptor,
};
pub use batch::{Batch, BatchExecutor, BatchedCircuit, SimpleExecutor};
pub use bridge::GateIdConversionError;
pub use canonicalizer::{CanonicalForm, GateCanonicalizer};
pub use decomposition::{
    AngleSource, CircuitResolver, DecompEntry, DecompOp, Decomposition, DecompositionRegistry,
    InstantiatedOp, Resolution, ResolutionError, ResolvedCircuit, ResolvedOp,
};
pub use definitions::{
    GateDefinitions, GateDefinitionsBuilder, GateDefinitionsError, GateExecutor, NoNativeGates,
};
pub use gate_id::{GateId, gates};
pub use gate_spec::{GateCategory, GateSpec};
pub use noise_integration::{DecompositionNoiseStrategy, GateIdNoiseConfig, GateNoiseParams};
pub use op_builder::{ConversionError, GateLibrary, OpBuilder, Subcircuit};
pub use operation::{
    AdaptedOp, AdaptedSequence, AncillaRequirements, ConditionalOp, MeasBasis, PrepBasis, ResultId,
};
pub use pauli::{Pauli, PauliString, StabilizerMeasurement, StabilizerPreparation};
pub use plugin::{
    CoreGatesPlugin, ExtendedDecompositionsPlugin, GatePlugin, PluginError, PluginLoader,
    StandardDecompositionsPlugin,
};
pub use queue_validation::{
    CommandQueueValidation, is_clifford_angle, is_clifford_circuit, is_clifford_gate_type,
    snap_command_queue,
};
pub use registry::GateRegistry;
pub use snapper::{AngleSnapper, SnapError, SnapPolicy, SnapResult};
pub use stabilizer_adaptor::{
    ExtendedAdaptor, StabilizerAdaptor, StabilizerMeasurementAdaptor, StabilizerPreparationAdaptor,
    stabilizer_gates,
};
pub use support_set::GateSupportSet;
pub use user_gates::{UserGateBuilder, UserGateDefinition, UserGateRegistry, UserGatesPlugin};
pub use validator::{
    AllowListValidator, CircuitValidator, CliffordTValidator, CliffordValidator,
    CompositeValidator, ExactAngleValidator, GateForValidation, ValidationError,
};

#[cfg(test)]
mod tests;
