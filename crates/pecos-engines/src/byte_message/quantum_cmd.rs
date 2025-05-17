use crate::core::result_id::ResultId;
use pecos_core::QubitId;
use std::fmt;

/// Structured command type for binary representation of quantum operations
///
/// This enum represents the various quantum operations that can be executed
/// by the quantum system, including gate operations, measurements, and
/// non-gate operations like record and message commands.
///
/// # Gate Operations
///
/// Gate operations represent quantum gates that are applied to qubits:
///
/// - `H` - Hadamard gate
/// - `X`, `Y`, `Z` - Pauli gates
/// - `CX` - Controlled-NOT gate
/// - `RZ` - Rotation around Z-axis
/// - `R1XY` - Rotation in XY plane
/// - `SZZ` - SZZ gate
/// - `RZZ` - Rotation around ZZ
///
/// # Measurement Operations
///
/// - `Measure` - Measure a qubit and store the result
/// - `Prep` - Prepare a qubit in the |0⟩ state
///
/// # Non-Gate Operations
///
/// - `Record` - Record data for classical processing
/// - `Message` - Send a message (info, warning, error, debug)
/// - `RecordResult` - Record a measurement result with a name for output
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumCmd {
    /// Hadamard gate on qubit
    H(QubitId),

    /// X gate (NOT gate) on qubit
    X(QubitId),

    /// Y gate on qubit
    Y(QubitId),

    /// Z gate on qubit
    Z(QubitId),

    /// CNOT gate with control and target qubits
    CX(QubitId, QubitId),

    /// RZ gate with angle (in radians) and qubit
    RZ(f64, QubitId),

    /// SZZ gate with two qubits
    SZZ(QubitId, QubitId),

    /// RZZ gate with angle (in radians) and two qubits
    RZZ(f64, QubitId, QubitId),

    /// Measure qubit and store in `result_id`
    Measure(QubitId, ResultId),

    /// Prepare qubit in the |0⟩ state
    Prep(QubitId),

    /// Record command with string data
    Record(String),

    /// Message command with string data
    Message(String),

    /// Records a result with a name for output
    ///
    /// This variant is used to associate a result ID with a name for output purposes.
    /// The first parameter is the result ID, and the second parameter is the name.
    RecordResult(ResultId, String),

    /// R1XY gate with theta, phi angles (in radians) and qubit
    R1XY(f64, f64, QubitId),

    /// U gate with theta, phi, lambda angles (in radians) and qubit
    U(f64, f64, f64, QubitId),
}

impl fmt::Display for QuantumCmd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantumCmd::H(qubit) => write!(f, "H {qubit}"),
            QuantumCmd::X(qubit) => write!(f, "X {qubit}"),
            QuantumCmd::Y(qubit) => write!(f, "Y {qubit}"),
            QuantumCmd::Z(qubit) => write!(f, "Z {qubit}"),
            QuantumCmd::CX(control, target) => write!(f, "CX {control} {target}"),
            QuantumCmd::RZ(theta, qubit) => write!(f, "RZ {qubit} {theta}"),
            QuantumCmd::SZZ(qubit1, qubit2) => write!(f, "SZZ {qubit1} {qubit2}"),
            QuantumCmd::RZZ(theta, qubit1, qubit2) => write!(f, "RZZ {qubit1} {qubit2} {theta}"),
            QuantumCmd::Measure(qubit, result) => write!(f, "M {qubit} {result}"),
            QuantumCmd::Prep(qubit) => write!(f, "Prep {qubit}"),
            QuantumCmd::Record(cmd) | QuantumCmd::Message(cmd) => write!(f, "{cmd}"),
            QuantumCmd::RecordResult(result, name) => write!(f, "RecordResult {result} {name}"),
            QuantumCmd::R1XY(theta, phi, qubit) => write!(f, "R1XY {theta} {phi} {qubit}"),
            QuantumCmd::U(theta, phi, lambda, qubit) => {
                write!(f, "U {theta} {phi} {lambda} {qubit}")
            }
        }
    }
}
