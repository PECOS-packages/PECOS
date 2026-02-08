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

mod gate_id;
mod gate_spec;
mod registry;
mod canonicalizer;
mod support_set;
mod snapper;
mod validator;
mod adaptor;
mod bridge;
mod queue_validation;
mod operation;
mod stabilizer_adaptor;
mod pauli;
mod op_builder;
#[macro_use]
mod circuit_macro;
mod decomposition;
mod plugin;
mod batch;
mod noise_integration;
mod user_gates;

pub use gate_id::{GateId, gates};
pub use gate_spec::{GateSpec, GateCategory};
pub use registry::GateRegistry;
pub use canonicalizer::{GateCanonicalizer, CanonicalForm};
pub use support_set::GateSupportSet;
pub use snapper::{AngleSnapper, SnapResult, SnapError, SnapPolicy};
pub use validator::{
    CircuitValidator, ValidationError, GateForValidation,
    CliffordValidator, CliffordTValidator, ExactAngleValidator,
    AllowListValidator, CompositeValidator,
};
pub use adaptor::{
    GateAdaptor, AdaptedGate, StandardAdaptor, CompositeAdaptor, CustomAdaptor,
    LiftedAdaptor, CompositeExtendedAdaptor,
};
pub use bridge::GateIdConversionError;
pub use queue_validation::{
    CommandQueueValidation, snap_command_queue, is_clifford_circuit,
    is_clifford_gate_type, is_clifford_angle,
};
pub use operation::{
    AdaptedOp, AdaptedSequence, AncillaRequirements, ResultId,
    PrepBasis, MeasBasis,
};
pub use stabilizer_adaptor::{
    ExtendedAdaptor, StabilizerAdaptor, StabilizerMeasurementAdaptor,
    StabilizerPreparationAdaptor, stabilizer_gates,
};
pub use pauli::{
    Pauli, PauliString, StabilizerMeasurement, StabilizerPreparation,
};
pub use op_builder::{OpBuilder, Subcircuit, GateLibrary, ConversionError};
pub use decomposition::{
    Decomposition, DecompositionRegistry, DecompEntry, DecompOp,
    AngleSource, InstantiatedOp, Resolution, ResolutionError,
    ResolvedCircuit, ResolvedOp, CircuitResolver,
};
pub use plugin::{
    GatePlugin, CoreGatesPlugin, StandardDecompositionsPlugin,
    ExtendedDecompositionsPlugin, PluginLoader, PluginError,
};
pub use batch::{
    Batch, BatchedCircuit, BatchExecutor, SimpleExecutor,
};
pub use noise_integration::{
    GateNoiseParams, GateIdNoiseConfig, DecompositionNoiseStrategy,
};
pub use user_gates::{
    UserGateBuilder, UserGateDefinition, UserGatesPlugin, UserGateRegistry,
};

#[cfg(test)]
mod tests;
