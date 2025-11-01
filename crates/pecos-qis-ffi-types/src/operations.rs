//! Quantum operation definitions
//!
//! This module defines the quantum operations that can be collected by the interface
//! and later executed by a runtime.

/// High-level quantum operations that include both QIS and control flow
#[derive(
    Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode,
)]
pub enum Operation {
    /// Quantum gate operation
    Quantum(QuantumOp),

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
#[derive(
    Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize, bincode::Encode, bincode::Decode,
)]
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
    Measure(usize, usize), // qubit, result_id

    // Reset
    Reset(usize),
}

impl From<QuantumOp> for Operation {
    fn from(op: QuantumOp) -> Self {
        Operation::Quantum(op)
    }
}
