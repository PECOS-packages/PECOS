//! Utilities for debugging and inspecting byte protocol messages
//!
//! This module provides functions for dumping and analyzing binary
//! messages for debugging purposes.

use crate::byte_message::message::ByteMessage;
use crate::byte_message::protocol::{
    BATCH_MAGIC, BatchHeader, GateHeader, MessageHeader, calc_padding,
};
use bytemuck;
use std::fmt::Write;
use std::io::Write as IoWrite;
use std::mem::size_of;

// ByteMessage guarantees 4-byte alignment by storing data in Vec<u32>

/// Dump a binary message batch to a string for debugging using modern structured parsing
#[must_use]
pub fn dump_batch(data: &[u8]) -> String {
    let mut output = String::new();

    // Try to parse as a structured ByteMessage first
    let message = ByteMessage::new(data);
    output.push_str("=== Structured ByteMessage Debug ===\n");

    // Determine message type
    match message.message_type() {
        Ok(msg_type) => {
            writeln!(output, "Message Type: {msg_type:?}").unwrap();
        }
        Err(e) => {
            writeln!(output, "Error determining message type: {e}").unwrap();
        }
    }

    // Try to parse quantum operations
    match message.quantum_ops() {
        Ok(operations) => {
            writeln!(output, "Quantum Operations ({} total):", operations.len()).unwrap();
            for (i, op) in operations.iter().enumerate() {
                writeln!(
                    output,
                    "  {i}: {} on qubits {:?} with params {:?}",
                    op.gate_type, op.qubits, op.params
                )
                .unwrap();
                writeln!(
                    output,
                    "      Classical arity: {}, Quantum arity: {}",
                    op.classical_arity(),
                    op.quantum_arity()
                )
                .unwrap();
            }
        }
        Err(e) => {
            writeln!(output, "No quantum operations (or error): {e}").unwrap();
        }
    }

    // Try to parse measurements
    match message.outcomes() {
        Ok(measurements) => {
            if !measurements.is_empty() {
                writeln!(
                    output,
                    "Measurement Results ({} total):",
                    measurements.len()
                )
                .unwrap();
                for (i, result) in measurements.iter().enumerate() {
                    writeln!(output, "  {i}: {result}").unwrap();
                }
            }
        }
        Err(e) => {
            writeln!(output, "No measurements (or error): {e}").unwrap();
        }
    }

    output.push_str("\n=== Raw Byte Analysis ===\n");

    // Append the original raw byte analysis for completeness
    output.push_str(&dump_batch_raw(data));
    output
}

/// Original raw byte dumping function for low-level debugging
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn dump_batch_raw(data: &[u8]) -> String {
    let mut output = String::new();

    // Check if we have enough bytes for a batch header
    if data.len() < size_of::<BatchHeader>() {
        output.push_str("ERROR: Data too small for batch header\n");
        return output;
    }

    // Parse batch header - unaligned read for external data compatibility
    let header = bytemuck::pod_read_unaligned::<BatchHeader>(&data[0..size_of::<BatchHeader>()]);

    if header.magic != BATCH_MAGIC {
        writeln!(
            output,
            "ERROR: Invalid magic number: 0x{:08x} (expected 0x{:08x})",
            header.magic, BATCH_MAGIC
        )
        .unwrap();
        return output;
    }

    output.push_str("Batch Header:\n");
    writeln!(output, "  Magic: 0x{:08x}", header.magic).unwrap();
    writeln!(output, "  Version: {}", header.version).unwrap();
    writeln!(output, "  Flags: 0x{:02x}", header.flags).unwrap();
    writeln!(output, "  Message Count: {}", header.msg_count).unwrap();
    writeln!(output, "  Total Size: {} bytes", header.total_size).unwrap();
    output.push('\n');

    // Parse each message
    let mut offset = size_of::<BatchHeader>();
    for i in 0..header.msg_count {
        // Ensure we have enough data for a message header
        if offset + size_of::<MessageHeader>() > data.len() {
            writeln!(output, "ERROR: Data too small for message {i} header").unwrap();
            break;
        }

        // Parse message header
        let msg_header = bytemuck::pod_read_unaligned::<MessageHeader>(
            &data[offset..offset + size_of::<MessageHeader>()],
        );

        offset += size_of::<MessageHeader>();

        // Get message type
        let msg_type = match msg_header.msg_type {
            10 => "Gate",
            20 => "Outcome",
            _ => "Unknown",
        };

        writeln!(output, "Message {i}:").unwrap();
        writeln!(output, "  Type: {} ({})", msg_type, msg_header.msg_type).unwrap();
        writeln!(output, "  Flags: 0x{:02x}", msg_header.flags).unwrap();
        writeln!(output, "  Payload Size: {} bytes", msg_header.payload_size).unwrap();

        // Parse payload based on message type
        if msg_header.payload_size > 0 {
            let payload_end = offset + msg_header.payload_size as usize;
            if payload_end > data.len() {
                writeln!(output, "ERROR: Data too small for message {i} payload").unwrap();
                break;
            }

            let payload = &data[offset..payload_end];

            match msg_header.msg_type {
                10 => {
                    // Gate (includes all gate operations including measurements)
                    if payload.len() >= size_of::<GateHeader>() {
                        let gate_header = bytemuck::pod_read_unaligned::<GateHeader>(
                            &payload[0..size_of::<GateHeader>()],
                        );

                        let gate_type = match std::panic::catch_unwind(|| {
                            pecos_core::gate_type::GateType::from(gate_header.gate_type)
                        }) {
                            Ok(gt) => format!("{gt}"),
                            Err(_) => format!("Unknown({})", gate_header.gate_type),
                        };

                        output.push_str("  Quantum Gate:\n");
                        writeln!(
                            output,
                            "    Type: {} ({})",
                            gate_type, gate_header.gate_type
                        )
                        .unwrap();
                        writeln!(output, "    Qubits: {}", gate_header.num_qubits).unwrap();
                        writeln!(
                            output,
                            "    Has Parameters: {}",
                            gate_header.has_params != 0
                        )
                        .unwrap();

                        // Dump qubit indices
                        let qubits_offset = size_of::<GateHeader>();
                        let mut qubits = Vec::new();

                        for i in 0..gate_header.num_qubits as usize {
                            let offset = qubits_offset + i * size_of::<u32>();
                            if offset + size_of::<u32>() <= payload.len() {
                                let qubit = u32::from_le_bytes([
                                    payload[offset],
                                    payload[offset + 1],
                                    payload[offset + 2],
                                    payload[offset + 3],
                                ]);
                                qubits.push(qubit);
                            }
                        }

                        writeln!(output, "    Qubit Indices: {qubits:?}").unwrap();

                        // Dump parameters if present
                        if gate_header.has_params != 0 {
                            let params_offset =
                                qubits_offset + gate_header.num_qubits as usize * size_of::<u32>();

                            match gate_header.gate_type {
                                32 => {
                                    // RZ
                                    if params_offset + size_of::<f64>() <= payload.len() {
                                        let theta = f64::from_le_bytes([
                                            payload[params_offset],
                                            payload[params_offset + 1],
                                            payload[params_offset + 2],
                                            payload[params_offset + 3],
                                            payload[params_offset + 4],
                                            payload[params_offset + 5],
                                            payload[params_offset + 6],
                                            payload[params_offset + 7],
                                        ]);

                                        writeln!(output, "    Theta: {theta}").unwrap();
                                    }
                                }
                                36 => {
                                    // R1XY
                                    if params_offset + 2 * size_of::<f64>() <= payload.len() {
                                        let phi = f64::from_le_bytes([
                                            payload[params_offset],
                                            payload[params_offset + 1],
                                            payload[params_offset + 2],
                                            payload[params_offset + 3],
                                            payload[params_offset + 4],
                                            payload[params_offset + 5],
                                            payload[params_offset + 6],
                                            payload[params_offset + 7],
                                        ]);

                                        let theta = f64::from_le_bytes([
                                            payload[params_offset + 8],
                                            payload[params_offset + 9],
                                            payload[params_offset + 10],
                                            payload[params_offset + 11],
                                            payload[params_offset + 12],
                                            payload[params_offset + 13],
                                            payload[params_offset + 14],
                                            payload[params_offset + 15],
                                        ]);

                                        writeln!(output, "    Phi: {phi}").unwrap();
                                        writeln!(output, "    Theta: {theta}").unwrap();
                                    }
                                }
                                _ => {}
                            }
                        }
                    }
                }
                20 => {
                    // MeasurementResult - use modern structured parsing
                    let message = ByteMessage::new(data);
                    match message.outcomes() {
                        Ok(measurements) => {
                            output.push_str("  Measurement Results:\n");
                            for (i, measurement) in measurements.iter().enumerate() {
                                writeln!(output, "    Result {i}: {measurement}").unwrap();
                            }
                        }
                        Err(e) => {
                            writeln!(output, "  Error parsing measurements: {e}").unwrap();
                        }
                    }
                }
                _ => {
                    // Dump raw bytes for other message types
                    writeln!(output, "  Payload: {payload:?}").unwrap();
                }
            }

            offset = payload_end;
        }

        // Skip padding to next message
        let padding = calc_padding(msg_header.payload_size as usize, 4);
        if padding > 0 {
            offset += padding;
        }

        output.push('\n');
    }

    output
}

/// Dump a `ByteMessage` to a string for debugging
#[must_use]
pub fn dump_message(message: &ByteMessage) -> String {
    dump_batch(message.as_bytes())
}

/// Utility function to write a `ByteMessage` to a file for debugging
///
/// # Errors
///
/// Returns an error if the file cannot be created or written to.
pub fn write_message_to_file(message: &ByteMessage, filename: &str) -> std::io::Result<()> {
    let mut file = std::fs::File::create(filename)?;
    file.write_all(message.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Gate;
    use crate::byte_message::ByteMessage;
    use pecos_core::Angle64;

    #[test]
    fn test_bytemap_dump() {
        // Create commands
        let commands = vec![Gate::h(&[0]), Gate::cx(&[(0, 1)])];

        // Create ByteMessage using the builder pattern
        let message = ByteMessage::builder().add_gate_commands(&commands).build();

        // Dump the message
        let dump = dump_message(&message);
        println!("{dump}");

        // Verify the dump contains expected information
        assert!(dump.contains("Batch Header"));
        assert!(dump.contains("Quantum Gate"));
    }

    #[test]
    fn test_dump_batch() {
        // Create a ByteMessage with different gate types
        let commands = vec![Gate::h(&[0]), Gate::rz(Angle64::from_radians(0.5), &[1])];

        // Create a ByteMessage using the builder
        let message = ByteMessage::builder().add_gate_commands(&commands).build();

        // Dump batch
        let dump = dump_batch(message.as_bytes());

        // Verify dump contains expected information
        assert!(dump.contains("Batch Header"));
        assert!(dump.contains("Magic: 0x5045"));
        assert!(dump.contains("Type: Gate"));
        assert!(dump.contains("Type: H"));
        assert!(dump.contains("Type: RZ"));
        assert!(dump.contains("Theta: 0.5"));
    }
}
