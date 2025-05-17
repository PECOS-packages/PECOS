use crate::common::get_thread_id;
use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::byte_message::ByteMessage;
use pecos_engines::byte_message::QuantumCmd;
use pecos_engines::byte_message::QuantumCommand;
use pecos_engines::byte_message::message_data::MessageData;
use pecos_engines::core::record_data::RecordData;

/// Parses binary commands from the QIR runtime into `QuantumCommand` objects
///
/// This function converts the binary commands from the QIR runtime into
/// `QuantumCommand` objects that can be processed by the quantum system.
///
/// # Arguments
///
/// * `commands` - The binary commands from the QIR runtime
///
/// # Returns
///
/// * `Vec<QuantumCommand>` - The parsed quantum commands
#[must_use]
pub fn parse_binary_commands(commands: &[QuantumCmd]) -> Vec<QuantumCommand> {
    QuantumCommand::parse_binary_commands(commands, |cmd| match cmd {
        QuantumCmd::H(qubit) => QuantumCommand::H(*qubit),
        QuantumCmd::X(qubit) => QuantumCommand::X(*qubit),
        QuantumCmd::Y(qubit) => QuantumCommand::Y(*qubit),
        QuantumCmd::Z(qubit) => QuantumCommand::Z(*qubit),
        QuantumCmd::CX(control, target) => QuantumCommand::CX(*control, *target),
        QuantumCmd::RZ(angle, qubit) => QuantumCommand::RZ(*angle, *qubit),
        QuantumCmd::R1XY(theta, phi, qubit) => QuantumCommand::R1XY(*theta, *phi, *qubit),
        QuantumCmd::U(theta, phi, lambda, qubit) => {
            QuantumCommand::U(*theta, *phi, *lambda, *qubit)
        }
        QuantumCmd::SZZ(qubit1, qubit2) => QuantumCommand::SZZ(*qubit1, *qubit2),
        QuantumCmd::RZZ(angle, qubit1, qubit2) => QuantumCommand::RZZ(*angle, *qubit1, *qubit2),
        QuantumCmd::Measure(qubit, result_id) => QuantumCommand::Measure(*qubit, *result_id),
        QuantumCmd::Prep(qubit) => QuantumCommand::Prep(*qubit),
        QuantumCmd::RecordResult(result_id, name) => {
            // Create a result record with the given name
            QuantumCommand::Record(RecordData::ResultRecord(result_id.0, Some(name.clone())))
        }
        QuantumCmd::Record(cmd) => {
            // Parse record commands into structured data
            let parts: Vec<&str> = cmd.split_whitespace().collect();
            if parts.len() >= 2 && parts[0] == "RECORD" {
                if let Ok(result_id) = parts[1].parse::<usize>() {
                    // This is a result record
                    let label = if parts.len() >= 3 {
                        Some(parts[2].to_string())
                    } else {
                        None
                    };
                    QuantumCommand::Record(RecordData::result(result_id, label))
                } else if parts.len() >= 3 {
                    // Try to parse as a key-value record
                    if let Ok(value) = parts[2].parse::<f64>() {
                        QuantumCommand::Record(RecordData::key_value(parts[1].to_string(), value))
                    } else {
                        // Fall back to raw record
                        debug!(
                            "QIR: Unable to parse record command as structured data: {}",
                            cmd
                        );
                        QuantumCommand::Record(RecordData::RawRecord(cmd.clone()))
                    }
                } else {
                    // Fall back to raw record
                    debug!(
                        "QIR: Unable to parse record command as structured data: {}",
                        cmd
                    );
                    QuantumCommand::Record(RecordData::RawRecord(cmd.clone()))
                }
            } else {
                // Fall back to raw record
                debug!(
                    "QIR: Unable to parse record command as structured data: {}",
                    cmd
                );
                QuantumCommand::Record(RecordData::RawRecord(cmd.clone()))
            }
        }
        QuantumCmd::Message(msg) => {
            // Parse message commands into structured data
            let msg_str = msg.as_str();
            if let Some(stripped) = msg_str.strip_prefix("Info: ") {
                QuantumCommand::Message(MessageData::info(stripped.to_string()))
            } else if let Some(stripped) = msg_str.strip_prefix("Warning: ") {
                QuantumCommand::Message(MessageData::warning(stripped.to_string()))
            } else if let Some(stripped) = msg_str.strip_prefix("Error: ") {
                QuantumCommand::Message(MessageData::error(stripped.to_string()))
            } else if let Some(stripped) = msg_str.strip_prefix("Debug: ") {
                QuantumCommand::Message(MessageData::debug(stripped.to_string()))
            } else {
                debug!(
                    "QIR: Unable to parse message command as structured data: {}",
                    msg
                );
                QuantumCommand::Message(MessageData::Raw(msg.clone()))
            }
        }
    })
}

/// Identifies circuit boundaries by analyzing command patterns
///
/// This function identifies circuit boundaries by looking for measurement patterns
/// in the command sequence. It helps determine which commands should be included
/// in a single circuit execution.
///
/// # Arguments
///
/// * `commands` - The quantum commands to analyze
///
/// # Returns
///
/// * `Vec<QuantumCommand>` - The commands up to the identified circuit boundary
#[must_use]
pub fn identify_circuit_boundaries(commands: &[QuantumCommand]) -> Vec<QuantumCommand> {
    // Identify circuit boundaries by looking for measurement patterns
    let mut measurement_indices = Vec::new();
    let mut gate_indices = Vec::new();

    // First pass: identify measurements and gates
    for (i, cmd) in commands.iter().enumerate() {
        if cmd.is_measurement() {
            measurement_indices.push(i);
        } else if cmd.is_gate() {
            gate_indices.push(i);
        }
    }

    // If we have both gates and measurements, try to find a circuit boundary
    if !gate_indices.is_empty() && !measurement_indices.is_empty() {
        // Find the first set of consecutive measurements at the end of a sequence of gates
        let last_gate_index = *gate_indices.last().unwrap_or(&0);
        let first_measurement_after_gates = measurement_indices
            .iter()
            .find(|&&idx| idx > last_gate_index);

        if let Some(&first_meas_idx) = first_measurement_after_gates {
            // Find consecutive measurements
            let mut end_idx = first_meas_idx;
            for (i, cmd) in commands.iter().enumerate().skip(first_meas_idx) {
                if cmd.is_measurement() {
                    end_idx = i;
                } else {
                    break;
                }
            }

            // Take all commands up to and including the last consecutive measurement
            let circuit_commands = commands[0..=end_idx].to_vec();
            debug!(
                "QIR: Found circuit boundary after {} commands with {} measurements",
                end_idx + 1,
                end_idx - first_meas_idx + 1
            );
            circuit_commands
        } else {
            // If we can't find measurements after gates, take all commands
            commands.to_vec()
        }
    } else {
        // If we don't have both gates and measurements, take all commands
        commands.to_vec()
    }
}

/// Converts a list of `QuantumCommands` to a `ByteMessage`
///
/// This function converts a list of `QuantumCommands` to a `ByteMessage` that can
/// be processed by the quantum system.
///
/// # Arguments
///
/// * `commands` - The quantum commands to convert
///
/// # Returns
///
/// * `Result<ByteMessage, PecosError>` - The `ByteMessage` if successful, or an error if the operation fails
pub fn commands_to_byte_message(commands: &[QuantumCommand]) -> Result<ByteMessage, PecosError> {
    // Get the current thread ID for logging
    let thread_id = get_thread_id();

    debug!(
        "QIR: [Thread {}] Converting {} commands to ByteMessage",
        thread_id,
        commands.len()
    );

    // Use the QuantumCommand's built-in method to convert to ByteMessage
    QuantumCommand::commands_to_byte_message(commands).map_err(|e| {
        PecosError::Processing(format!("Failed to convert commands to ByteMessage: {e}"))
    })
}
