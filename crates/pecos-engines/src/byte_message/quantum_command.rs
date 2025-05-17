use crate::byte_message::gate_type::GateType;
use crate::byte_message::message_data::MessageData;
use crate::byte_message::protocol::{MessageFlags, MessageType};
use crate::byte_message::{ByteMessage, ByteMessageBuilder};
use crate::core::record_data::RecordData;
use crate::core::result_id::ResultId;
use log::debug;
use pecos_core::QubitId;
use pecos_core::errors::PecosError;
use std::fmt;

/// Command type for unknown commands
///
/// This enum represents the various types of unknown commands that can be
/// encountered during program execution. It helps categorize unknown
/// commands for better error reporting and debugging.
#[derive(Debug, Clone, PartialEq)]
pub enum CommandType {
    /// Unknown gate command
    Gate(String),

    /// Unknown control command
    Control(String),

    /// Other unknown command
    Other(String),
}

impl fmt::Display for CommandType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CommandType::Gate(cmd) => write!(f, "Unknown Gate: {cmd}"),
            CommandType::Control(cmd) => write!(f, "Unknown Control: {cmd}"),
            CommandType::Other(cmd) => write!(f, "Unknown Command: {cmd}"),
        }
    }
}

/// Represents a quantum command with its parameters
///
/// This enum represents the various quantum operations that can be executed
/// by the quantum system, including gate operations, measurements, and
/// non-gate operations like record and message commands.
#[derive(Debug, Clone, PartialEq)]
pub enum QuantumCommand {
    // Gate operations
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

    /// R1XY gate with two angles (in radians) and qubit
    R1XY(f64, f64, QubitId),

    /// U gate with three angles (in radians) and qubit
    U(f64, f64, f64, QubitId),

    /// SZZ gate with two qubits
    SZZ(QubitId, QubitId),

    /// RZZ gate with angle (in radians) and two qubits
    RZZ(f64, QubitId, QubitId),

    /// Measure qubit and store in `result_id`
    Measure(QubitId, ResultId),

    /// Prepare qubit in the |0⟩ state
    Prep(QubitId),

    // Non-gate operations (for filtering)
    /// Record command with structured data
    Record(RecordData),

    /// Message command with structured data
    Message(MessageData),

    /// Unknown command with command type
    Unknown(CommandType),
}

impl QuantumCommand {
    /// Parse commands directly from binary data
    /// This is a more efficient alternative to string-based parsing
    #[must_use]
    pub fn parse_binary_commands<T>(commands: &[T], parse_fn: impl Fn(&T) -> Self) -> Vec<Self> {
        commands.iter().map(parse_fn).collect()
    }

    /// Get the `GateType` for this command
    #[must_use]
    pub fn gate_type_id(&self) -> Option<GateType> {
        match self {
            QuantumCommand::H(_) => Some(GateType::H),
            QuantumCommand::X(_) => Some(GateType::X),
            QuantumCommand::Y(_) => Some(GateType::Y),
            QuantumCommand::Z(_) => Some(GateType::Z),
            QuantumCommand::CX(_, _) => Some(GateType::CX),
            QuantumCommand::RZ(_, _) => Some(GateType::RZ),
            QuantumCommand::R1XY(_, _, _) => Some(GateType::R1XY),
            QuantumCommand::U(_, _, _, _) => Some(GateType::U),
            QuantumCommand::SZZ(_, _) => Some(GateType::SZZ),
            QuantumCommand::RZZ(_, _, _) => Some(GateType::RZZ),
            QuantumCommand::Measure(_, _) => Some(GateType::Measure),
            QuantumCommand::Prep(_) => Some(GateType::Prep),
            _ => None,
        }
    }

    /// Check if this command is a gate operation
    #[must_use]
    pub fn is_gate(&self) -> bool {
        self.gate_type_id().is_some()
    }

    /// Check if this command is supported for quantum processing
    #[must_use]
    pub fn is_supported(&self) -> bool {
        self.is_gate()
    }

    /// Check if this command is a measurement
    #[must_use]
    pub fn is_measurement(&self) -> bool {
        matches!(self, QuantumCommand::Measure(_, _))
    }

    /// Get the `result_id` if this is a measurement command or a result record
    #[must_use]
    pub fn result_id(&self) -> Option<usize> {
        match self {
            QuantumCommand::Measure(_, result_id) => Some(result_id.0),
            QuantumCommand::Record(RecordData::ResultRecord(result_id, _)) => Some(*result_id),
            _ => None,
        }
    }

    /// Add this command directly to a `ByteMessageBuilder`
    pub fn add_to_builder(&self, builder: &mut ByteMessageBuilder) -> Result<(), PecosError> {
        match self {
            QuantumCommand::H(qubit) => {
                builder.add_h(&[qubit.0]);
                Ok(())
            }
            QuantumCommand::X(qubit) => {
                builder.add_x(&[qubit.0]);
                Ok(())
            }
            QuantumCommand::Y(qubit) => {
                builder.add_y(&[qubit.0]);
                Ok(())
            }
            QuantumCommand::Z(qubit) => {
                builder.add_z(&[qubit.0]);
                Ok(())
            }
            QuantumCommand::CX(control, target) => {
                builder.add_cx(&[control.0], &[target.0]);
                Ok(())
            }
            QuantumCommand::RZ(angle, qubit) => {
                builder.add_rz(*angle, &[qubit.0]);
                Ok(())
            }
            QuantumCommand::R1XY(theta, phi, qubit) => {
                builder.add_r1xy(*theta, *phi, &[qubit.0]);
                Ok(())
            }
            QuantumCommand::U(theta, phi, lambda, qubit) => {
                builder.add_u(*theta, *phi, *lambda, &[qubit.0]);
                Ok(())
            }
            QuantumCommand::SZZ(qubit1, qubit2) => {
                builder.add_szz(&[qubit1.0], &[qubit2.0]);
                Ok(())
            }
            QuantumCommand::RZZ(angle, qubit1, qubit2) => {
                builder.add_rzz(*angle, &[qubit1.0], &[qubit2.0]);
                Ok(())
            }
            QuantumCommand::Measure(qubit, result_id) => {
                builder.add_measurements(&[qubit.0], &[result_id.0]);
                Ok(())
            }
            QuantumCommand::Prep(qubit) => {
                builder.add_prep(&[qubit.0]);
                Ok(())
            }
            QuantumCommand::Record(data) => {
                match data {
                    RecordData::ResultRecord(result_id, label) => {
                        builder.add_result_record(*result_id, label.as_deref());
                        debug!("Added ResultRecord to ByteMessageBuilder");
                    }
                    RecordData::KeyValueRecord(key, value) => {
                        builder.add_record_data(key, *value);
                        debug!("Added KeyValueRecord to ByteMessageBuilder");
                    }
                    RecordData::RawRecord(cmd) => {
                        // For raw records, we still need to use the string representation
                        let payload = cmd.clone().into_bytes();
                        builder.add_message(MessageType::RecordData, &payload, MessageFlags::NONE);
                        debug!("Added RawRecord to ByteMessageBuilder");
                    }
                }
                Ok(())
            }
            QuantumCommand::Message(data) => {
                match data {
                    MessageData::Info(msg) => {
                        builder.add_info_message(msg);
                        debug!("Added Info message to ByteMessageBuilder");
                    }
                    MessageData::Warning(msg) => {
                        builder.add_warning_message(msg);
                        debug!("Added Warning message to ByteMessageBuilder");
                    }
                    MessageData::Error(msg) => {
                        builder.add_error_message(msg);
                        debug!("Added Error message to ByteMessageBuilder");
                    }
                    MessageData::Debug(msg) => {
                        builder.add_debug_message(msg);
                        debug!("Added Debug message to ByteMessageBuilder");
                    }
                    MessageData::Raw(msg) => {
                        // For raw messages, we still need to use the string representation
                        let payload = msg.clone().into_bytes();
                        builder.add_message(MessageType::InfoMessage, &payload, MessageFlags::NONE);
                        debug!("Added Raw message to ByteMessageBuilder");
                    }
                }
                Ok(())
            }
            QuantumCommand::Unknown(cmd) => {
                // For unknown commands, use an error message
                let error_msg = cmd.to_string();
                builder.add_error_message(&error_msg);
                debug!("Added Unknown command as error message to ByteMessageBuilder");
                Ok(())
            }
        }
    }

    /// Convert the command to a `ByteMessage`
    /// This is more efficient than string-based serialization for gate operations
    pub fn to_byte_message(&self) -> Result<ByteMessage, PecosError> {
        let mut builder = ByteMessage::quantum_operations_builder();
        self.add_to_builder(&mut builder)?;
        Ok(builder.build())
    }

    /// Convert a list of `QuantumCommands` to a `ByteMessage`
    /// This handles all command types, including gate operations, records, and messages
    pub fn commands_to_byte_message(commands: &[Self]) -> Result<ByteMessage, PecosError> {
        let mut builder = ByteMessage::quantum_operations_builder();

        for cmd in commands {
            cmd.add_to_builder(&mut builder)?;
        }

        Ok(builder.build())
    }
}

impl fmt::Display for QuantumCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantumCommand::H(qubit) => write!(f, "H {qubit}"),
            QuantumCommand::X(qubit) => write!(f, "X {qubit}"),
            QuantumCommand::Y(qubit) => write!(f, "Y {qubit}"),
            QuantumCommand::Z(qubit) => write!(f, "Z {qubit}"),
            QuantumCommand::CX(control, target) => write!(f, "CX {control} {target}"),
            QuantumCommand::RZ(angle, qubit) => write!(f, "RZ {angle} {qubit}"),
            QuantumCommand::R1XY(theta, phi, qubit) => write!(f, "R1XY {theta} {phi} {qubit}"),
            QuantumCommand::U(theta, phi, lambda, qubit) => {
                write!(f, "U {theta} {phi} {lambda} {qubit}")
            }
            QuantumCommand::SZZ(qubit1, qubit2) => write!(f, "SZZ {qubit1} {qubit2}"),
            QuantumCommand::RZZ(angle, qubit1, qubit2) => {
                write!(f, "RZZ {angle} {qubit1} {qubit2}")
            }
            QuantumCommand::Measure(qubit, result_id) => write!(f, "M {qubit} {result_id}"),
            QuantumCommand::Prep(qubit) => write!(f, "PREP {qubit}"),
            QuantumCommand::Record(data) => match data {
                RecordData::ResultRecord(result_id, Some(label)) => {
                    write!(f, "RECORD {result_id} {label}")
                }
                RecordData::ResultRecord(result_id, None) => write!(f, "RECORD {result_id}"),
                RecordData::KeyValueRecord(key, value) => write!(f, "RECORD {key} {value}"),
                RecordData::RawRecord(cmd) => write!(f, "{cmd}"),
            },
            QuantumCommand::Message(data) => match data {
                MessageData::Info(msg) => write!(f, "MESSAGE Info: {msg}"),
                MessageData::Warning(msg) => write!(f, "MESSAGE Warning: {msg}"),
                MessageData::Error(msg) => write!(f, "MESSAGE Error: {msg}"),
                MessageData::Debug(msg) => write!(f, "MESSAGE Debug: {msg}"),
                MessageData::Raw(msg) => write!(f, "{msg}"),
            },
            QuantumCommand::Unknown(cmd) => write!(f, "{cmd}"),
        }
    }
}
