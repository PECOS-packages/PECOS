//! Quantum operation definitions
//!
//! This module defines the quantum operations that can be collected by the interface
//! and later executed by a runtime.

use std::collections::BTreeMap;

/// Structured metadata attached to QIS operations or lowered quantum operations.
pub type TraceMetadata = BTreeMap<String, String>;

/// Runtime provenance for a named `result(...)` output.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NamedResultTrace {
    /// Name passed to `result(name, value)`.
    pub name: String,
    /// Boolean values emitted for this result call.
    pub values: Vec<bool>,
    /// Runtime measurement result IDs read to produce `values`, in element order.
    pub result_ids: Vec<usize>,
}

/// High-level quantum operations that include both QIS and control flow
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum Operation {
    /// Quantum gate operation
    Quantum(QuantumOp),

    /// Source-level metadata intended to annotate subsequent lowered operations.
    ///
    /// Runtimes may preserve this metadata when lowering, scheduling, or expanding
    /// operations. PECOS's direct lowering path attaches it to the next emitted
    /// simulator gate and then clears it.
    TraceMetadata {
        metadata: TraceMetadata,
        /// Optional source qubit that owns this metadata.
        ///
        /// Runtime lowering uses this to wait for the next compatible source
        /// operation touching the same qubit, which is stricter than attaching
        /// metadata to the next operation in global program order.
        #[serde(default)]
        qubit: Option<usize>,
    },

    /// Allocate a qubit
    AllocateQubit { id: usize },

    /// Allocate a result slot
    AllocateResult { id: usize },

    /// Release a qubit
    ReleaseQubit { id: usize },

    /// Record output mapping from result ID to classical register name
    RecordOutput {
        result_id: usize,
        register_name: String,
    },

    /// Classical control flow marker
    Barrier,
}

/// Quantum operations that can be executed
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum QuantumOp {
    // Single-qubit gates
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    S(usize),
    Sdg(usize),
    T(usize),
    Tdg(usize),

    // Rotation gates
    RX(f64, usize),
    RY(f64, usize),
    RZ(f64, usize),

    // Hardware-native gates (for Selene compatibility)
    RXY(f64, f64, usize), // theta, phi, qubit

    // Idle period in seconds for time-based noise models
    Idle(f64, usize), // duration_seconds, qubit

    // Two-qubit gates
    CX(usize, usize),
    CY(usize, usize),
    CZ(usize, usize),
    CH(usize, usize),

    // Controlled rotations
    CRZ(f64, usize, usize),

    // Three-qubit gates
    CCX(usize, usize, usize),

    // ZZ interaction
    ZZ(usize, usize),
    RZZ(f64, usize, usize),

    // Measurement
    Measure(usize, usize),       // qubit, result_id
    MeasureLeaked(usize, usize), // qubit, result_id; outcome is 0, 1, or 2

    // Reset
    Reset(usize),
}

/// A lowered quantum operation plus any provenance supplied by the lowering runtime.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct LoweredQuantumOp {
    /// Lowered operation to send to the quantum/noise engine.
    pub op: QuantumOp,
    /// Generic trace/source metadata associated with this lowered operation.
    pub metadata: TraceMetadata,
}

impl LoweredQuantumOp {
    /// Create a lowered operation with explicit metadata.
    #[must_use]
    pub fn new(op: QuantumOp, metadata: TraceMetadata) -> Self {
        Self { op, metadata }
    }
}

impl From<QuantumOp> for LoweredQuantumOp {
    fn from(op: QuantumOp) -> Self {
        Self {
            op,
            metadata: TraceMetadata::new(),
        }
    }
}

impl From<QuantumOp> for Operation {
    fn from(op: QuantumOp) -> Self {
        Operation::Quantum(op)
    }
}
