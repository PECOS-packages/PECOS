//! Message builder for creating byte-encoded messages
//!
//! This module provides utilities for constructing binary messages
//! according to the byte protocol.

use crate::byte_message::message::ByteMessage;
use crate::byte_message::protocol::{
    BatchHeader, GateHeader, MessageFlags, MessageHeader, MessageType, OutcomeHeader,
    ReturnValueHeader, calc_padding,
};
use bytemuck::bytes_of;
use pecos_core::gates::Gate;
use pecos_core::{Angle64, QubitId};
use std::mem::size_of;

// ByteMessage guarantees 4-byte alignment by storing data in Vec<u32>

// TODO: Make add_gates() add multiple qubits at a single time...

/// Enum to track what kind of message is being built
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BuilderMode {
    Empty,               // No operations added yet
    QuantumOperations,   // Contains quantum operations
    MeasurementOutcomes, // Contains measurement outcomes
    ReturnValue,         // Contains return value
}

/// Helper for building binary messages
///
/// The builder maintains internal state tracking what kind of message is being created
/// and ensures that different message types are not mixed inappropriately.
#[derive(Debug)]
pub struct ByteMessageBuilder {
    buffer: Vec<u8>,
    msg_count: u32,
    mode: BuilderMode,
}

impl Default for ByteMessageBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for ByteMessageBuilder {
    fn clone(&self) -> Self {
        Self {
            buffer: self.buffer.clone(),
            msg_count: self.msg_count,
            mode: self.mode,
        }
    }
}

impl ByteMessageBuilder {
    /// Create a new message builder
    #[must_use]
    pub fn new() -> Self {
        let mut buffer = Vec::with_capacity(512); // Pre-allocate reasonable buffer
        // Reserve space for batch header, will fill later
        buffer.resize(size_of::<BatchHeader>(), 0);

        Self {
            buffer,
            msg_count: 0,
            mode: BuilderMode::Empty,
        }
    }

    /// Create a builder pre-configured for quantum operations
    ///
    /// Sets the builder mode to `QuantumOperations` to build a message
    /// containing quantum gates and operations.
    ///
    /// # Returns
    ///
    /// Returns `self` for method chaining.
    #[must_use]
    pub fn for_quantum_operations(&mut self) -> &mut Self {
        self.mode = BuilderMode::QuantumOperations;
        self
    }

    /// Create a builder pre-configured for measurement outcomes
    ///
    /// Sets the builder mode to `MeasurementOutcomes` to build a message
    /// containing measurement outcomes.
    ///
    /// # Returns
    ///
    /// Returns `self` for method chaining.
    #[must_use]
    pub fn for_outcomes(&mut self) -> &mut Self {
        self.mode = BuilderMode::MeasurementOutcomes;
        self
    }

    /// Add padding bytes to ensure alignment
    fn add_padding(&mut self, alignment: usize) {
        let padding = calc_padding(self.buffer.len(), alignment);
        if padding > 0 {
            self.buffer.resize(self.buffer.len() + padding, 0);
        }
    }

    /// Add a message with a header and payload
    ///
    /// This method adds a new message to the builder with the specified type, payload,
    /// and flags. It ensures proper alignment and maintains the builder's mode.
    ///
    /// # Arguments
    ///
    /// * `msg_type` - The type of message to add (`MessageType::Gate` or `MessageType::Outcome`)
    /// * `payload` - The binary payload for the message
    /// * `flags` - Optional flags to set on the message
    ///
    /// # Returns
    ///
    /// Returns `self` for method chaining.
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - Attempting to mix quantum operations and measurement outcomes in the same message
    pub fn add_message(
        &mut self,
        msg_type: MessageType,
        payload: &[u8],
        flags: MessageFlags,
    ) -> &mut Self {
        // Validate message type compatibility with current mode
        match msg_type {
            MessageType::Gate => {
                // Gates require QuantumOperations mode
                assert!(
                    !(self.mode == BuilderMode::MeasurementOutcomes),
                    "Cannot mix quantum operations and measurement outcomes in the same message"
                );

                // Auto-set mode if not already set
                if self.mode == BuilderMode::Empty {
                    self.mode = BuilderMode::QuantumOperations;
                }
            }
            MessageType::Outcome => {
                // Outcomes require MeasurementOutcomes mode
                assert!(
                    !(self.mode == BuilderMode::QuantumOperations
                        || self.mode == BuilderMode::ReturnValue),
                    "Cannot mix measurement outcomes with other message types"
                );

                // Always set the mode (even if already in Empty state)
                self.mode = BuilderMode::MeasurementOutcomes;
            }
            MessageType::ReturnValue => {
                // Return values should be sent separately
                assert!(
                    self.mode == BuilderMode::Empty || self.mode == BuilderMode::ReturnValue,
                    "Cannot mix return values with other message types"
                );
                self.mode = BuilderMode::ReturnValue;
            }
        }

        // Ensure 4-byte alignment for message header
        self.add_padding(4);

        // Create and write message header
        let payload_size = u32::try_from(payload.len()).unwrap_or_else(|_| {
            // This is a very unlikely case, but we handle it gracefully
            log::warn!("Payload size exceeds u32::MAX, using maximum value");
            u32::MAX
        });

        let header = MessageHeader::new(msg_type, payload_size, flags);
        self.buffer.extend_from_slice(bytes_of(&header));

        // Write payload
        self.buffer.extend_from_slice(payload);

        // Increment message count
        self.msg_count += 1;

        // Return self for method chaining
        self
    }

    /// Add a quantum gate command
    ///
    /// This method adds a quantum gate to the message builder.
    ///
    /// # Arguments
    ///
    /// * `gate` - The quantum gate to add
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    ///
    /// # Panics
    ///
    /// This function will panic if the number of qubits in the gate exceeds 255,
    /// as the protocol uses a u8 to represent the qubit count.
    pub fn add_gate_command(&mut self, gate: &Gate) -> &mut Self {
        // Calculate total payload size
        // Classical parameters in wire format include both angles (as radians) and other params
        let header_size = size_of::<GateHeader>();
        let qubits_size = gate.qubits.len() * size_of::<u32>();
        let params_size = (gate.angles.len() + gate.params.len()) * size_of::<f64>();
        let total_size = header_size + qubits_size + params_size;

        // Create a buffer for the payload
        let mut payload = Vec::with_capacity(total_size);

        // Determine if there are any parameters (angles or other params)
        let has_params = !gate.angles.is_empty() || !gate.params.is_empty();

        // Create gate header
        let header = GateHeader {
            gate_type: gate.gate_type as u8,
            num_qubits: u8::try_from(gate.qubits.len()).expect("Too many qubits for gate"),
            has_params: u8::from(has_params),
            reserved: 0,
        };

        // Add header to payload
        payload.extend_from_slice(bytes_of(&header));

        // Add qubit indices to payload (convert QubitId to usize to u32)
        for qubit in &gate.qubits {
            let qubit_u32 = u32::try_from(usize::from(*qubit)).expect("Qubit index too large");
            payload.extend_from_slice(&qubit_u32.to_le_bytes());
        }

        // Add angles to payload (converted to radians for wire format)
        for angle in &gate.angles {
            let radians = angle.to_radians();
            payload.extend_from_slice(&radians.to_le_bytes());
        }

        // Add other parameters to payload if any (e.g., duration for Idle)
        for param in &gate.params {
            payload.extend_from_slice(&param.to_le_bytes());
        }

        // Add the message to the buffer
        self.add_message(MessageType::Gate, &payload, MessageFlags::NONE);
        self
    }

    /// Add multiple gate commands at once
    pub fn add_gate_commands(&mut self, gates: &[Gate]) -> &mut Self {
        for gate in gates {
            self.add_gate_command(gate);
        }
        self
    }

    /// Add multiple measurement outcomes at once
    ///
    /// # Panics
    ///
    /// Panics if any result outcome is too large to fit in a u32.
    pub fn add_outcomes(&mut self, outcomes: &[usize]) -> &mut Self {
        for &result in outcomes {
            let result_header = OutcomeHeader {
                outcome: u32::try_from(result).expect("Result outcome too large"),
            };

            self.add_message(
                MessageType::Outcome,
                bytes_of(&result_header),
                MessageFlags::NONE,
            );
        }
        self
    }

    /// Add a return value from program execution
    ///
    /// This is typically used to send the return value from `teardown()`
    /// back to PECOS through the IPC channel.
    pub fn add_return_value(&mut self, value: i64) -> &mut Self {
        let return_header = ReturnValueHeader { value };

        self.add_message(
            MessageType::ReturnValue,
            bytes_of(&return_header),
            MessageFlags::NONE,
        );
        self
    }

    /// Add idle operations for specified qubits for a given duration
    ///
    /// # Arguments
    ///
    /// * `duration` - The duration of the idle period in seconds
    /// * `qubits` - The qubits that are idling
    ///
    /// # Returns
    ///
    /// A mutable reference to self for method chaining
    pub fn add_idle(&mut self, duration: f64, qubits: &[usize]) -> &mut Self {
        // Ensure we have qubits to work with
        if qubits.is_empty() {
            return self;
        }

        let mut idle_qubits = Vec::with_capacity(qubits.len());
        for &q in qubits {
            idle_qubits.push(q);
        }

        // Create and add the idle gate
        let idle_qubits_id: Vec<QubitId> = idle_qubits.into_iter().map(QubitId).collect();
        let gate = Gate::idle(duration, idle_qubits_id);
        self.add_gate_command(&gate)
    }

    /// Add an X gate
    pub fn add_x(&mut self, qubits: &[usize]) -> &mut Self {
        let gate = Gate::x(qubits);
        self.add_gate_command(&gate);
        self
    }

    /// Add a Y gate
    pub fn add_y(&mut self, qubits: &[usize]) -> &mut Self {
        let gate = Gate::y(qubits);
        self.add_gate_command(&gate);
        self
    }

    /// Add a Z gate
    pub fn add_z(&mut self, qubits: &[usize]) -> &mut Self {
        let gate = Gate::z(qubits);
        self.add_gate_command(&gate);
        self
    }

    /// Add an H gate
    pub fn add_h(&mut self, qubits: &[usize]) -> &mut Self {
        let gate = Gate::h(qubits);
        self.add_gate_command(&gate);
        self
    }

    /// Add CX (controlled-X) gates between pairs of qubits
    ///
    /// # Panics
    ///
    /// This function will panic if the controls and targets arrays do not have the same length.
    pub fn add_cx(&mut self, controls: &[usize], targets: &[usize]) -> &mut Self {
        assert_eq!(
            controls.len(),
            targets.len(),
            "Controls and targets arrays must have the same length"
        );
        let pairs: Vec<(usize, usize)> = controls
            .iter()
            .zip(targets.iter())
            .map(|(&c, &t)| (c, t))
            .collect();
        let gate = Gate::cx(&pairs);
        self.add_gate_command(&gate);
        self
    }

    /// Add RZZ gates between pairs of qubits
    ///
    /// # Panics
    ///
    /// This function will panic if the qubits1 and qubits2 arrays do not have the same length.
    pub fn add_rzz(&mut self, theta: Angle64, qubits1: &[usize], qubits2: &[usize]) -> &mut Self {
        assert_eq!(
            qubits1.len(),
            qubits2.len(),
            "Qubit1 and qubit2 arrays must have the same length"
        );
        let pairs: Vec<(usize, usize)> = qubits1
            .iter()
            .zip(qubits2.iter())
            .map(|(&q1, &q2)| (q1, q2))
            .collect();
        let gate = Gate::rzz(theta, &pairs);
        self.add_gate_command(&gate);
        self
    }

    /// Add SZZ gates between pairs of qubits
    ///
    /// # Panics
    ///
    /// This function will panic if the qubits1 and qubits2 arrays do not have the same length.
    pub fn add_szz(&mut self, qubits1: &[usize], qubits2: &[usize]) -> &mut Self {
        assert_eq!(
            qubits1.len(),
            qubits2.len(),
            "Qubit1 and qubit2 arrays must have the same length"
        );
        let pairs: Vec<(usize, usize)> = qubits1
            .iter()
            .zip(qubits2.iter())
            .map(|(&q1, &q2)| (q1, q2))
            .collect();
        let gate = Gate::szz(&pairs);
        self.add_gate_command(&gate);
        self
    }

    /// Add an `SZZdg` gate
    ///
    /// # Arguments
    ///
    /// * `qubits1` - First set of qubits
    /// * `qubits2` - Second set of qubits
    ///
    /// # Returns
    ///
    /// * `&mut Self` - Returns self for method chaining
    ///
    /// # Panics
    ///
    /// This function will panic if the qubits1 and qubits2 arrays do not have the same length.
    pub fn add_szzdg(&mut self, qubits1: &[usize], qubits2: &[usize]) -> &mut Self {
        assert_eq!(
            qubits1.len(),
            qubits2.len(),
            "Qubit1 and qubit2 arrays must have the same length"
        );
        let pairs: Vec<(usize, usize)> = qubits1
            .iter()
            .zip(qubits2.iter())
            .map(|(&q1, &q2)| (q1, q2))
            .collect();
        let gate = Gate::szzdg(&pairs);
        self.add_gate_command(&gate);
        self
    }

    /// Add an RZ gate
    pub fn add_rz(&mut self, theta: Angle64, qubits: &[usize]) -> &mut Self {
        let gate = Gate::rz(theta, qubits);
        self.add_gate_command(&gate);
        self
    }

    /// Add an R1XY gate
    pub fn add_r1xy(&mut self, theta: Angle64, phi: Angle64, qubits: &[usize]) -> &mut Self {
        let gate = Gate::r1xy(theta, phi, qubits);
        self.add_gate_command(&gate);
        self
    }

    /// Add a U gate
    pub fn add_u(
        &mut self,
        theta: Angle64,
        phi: Angle64,
        lambda: Angle64,
        qubits: &[usize],
    ) -> &mut Self {
        let gate = Gate::u(theta, phi, lambda, qubits);
        self.add_gate_command(&gate);
        self
    }

    /// Add measurement operations for multiple qubits
    ///
    /// # Panics
    ///
    /// Panics if any qubit ID is too large to fit in a u32.
    pub fn add_measurements(&mut self, qubit_ids: &[usize]) -> &mut Self {
        for &qubit in qubit_ids {
            // Add a measurement as a regular gate command
            let gate = Gate::measure(&[qubit]);
            self.add_gate_command(&gate);
        }
        self
    }

    /// Add measure leakage operations for multiple qubits
    ///
    /// This behaves like `add_measurements()` but is intended for measuring qubits
    /// that may be in a leaked state. In the future, this will output 0, 1, or 2
    /// (where 2 indicates the qubit is leaked).
    ///
    /// # Panics
    ///
    /// Panics if any qubit ID is too large to fit in a u32.
    pub fn add_measure_leakages(&mut self, qubit_ids: &[usize]) -> &mut Self {
        for &qubit in qubit_ids {
            // Add a measure_leaked as a regular gate command
            let gate = Gate::measure_leaked(&[qubit]);
            self.add_gate_command(&gate);
        }
        self
    }

    /// Add a `MeasCrosstalkGlobalPayload`
    pub fn add_meas_crosstalk_global_payload(&mut self, qubits: &[usize]) -> &mut Self {
        let gate = Gate::meas_crosstalk_global_payload(qubits);
        self.add_gate_command(&gate);
        self
    }

    /// Add a `MeasCrosstalkLocalPayload`
    pub fn add_meas_crosstalk_local_payload(&mut self, qubits: &[usize]) -> &mut Self {
        let gate = Gate::meas_crosstalk_local_payload(qubits);
        self.add_gate_command(&gate);
        self
    }

    /// Add a Prep gate
    pub fn add_prep(&mut self, qubits: &[usize]) -> &mut Self {
        let gate = Gate::prep(qubits);
        self.add_gate_command(&gate);
        self
    }

    /// Add an SZ (S) gate
    pub fn add_sz(&mut self, qubits: &[usize]) -> &mut Self {
        // S gate is RZ(π/2)
        self.add_rz(Angle64::QUARTER_TURN, qubits)
    }

    /// Add an `SZdg` (S†) gate
    pub fn add_szdg(&mut self, qubits: &[usize]) -> &mut Self {
        // S† gate is RZ(-π/2)
        self.add_rz(-Angle64::QUARTER_TURN, qubits)
    }

    /// Add a T gate
    pub fn add_t(&mut self, qubits: &[usize]) -> &mut Self {
        // T gate is RZ(π/4)
        self.add_rz(Angle64::QUARTER_TURN / 2u64, qubits)
    }

    /// Add a Tdg (T†) gate
    pub fn add_tdg(&mut self, qubits: &[usize]) -> &mut Self {
        // T† gate is RZ(-π/4)
        self.add_rz(-(Angle64::QUARTER_TURN / 2u64), qubits)
    }

    /// Add an RX gate
    pub fn add_rx(&mut self, theta: Angle64, qubits: &[usize]) -> &mut Self {
        let gate = Gate::rx(theta, qubits);
        self.add_gate_command(&gate);
        self
    }

    /// Add an RY gate
    pub fn add_ry(&mut self, theta: Angle64, qubits: &[usize]) -> &mut Self {
        let gate = Gate::ry(theta, qubits);
        self.add_gate_command(&gate);
        self
    }

    /// Add a CY gate
    ///
    /// # Panics
    ///
    /// Panics if the length of `controls` and `targets` are not equal.
    pub fn add_cy(&mut self, controls: &[usize], targets: &[usize]) -> &mut Self {
        // CY = (I ⊗ Sdg) CX (I ⊗ S)
        assert_eq!(
            controls.len(),
            targets.len(),
            "Controls and targets must have same length"
        );
        for (&c, &t) in controls.iter().zip(targets.iter()) {
            self.add_szdg(&[t]);
            self.add_cx(&[c], &[t]);
            self.add_sz(&[t]);
        }
        self
    }

    /// Add a CZ gate
    ///
    /// # Panics
    ///
    /// Panics if the length of `controls` and `targets` are not equal.
    pub fn add_cz(&mut self, controls: &[usize], targets: &[usize]) -> &mut Self {
        // CZ = H CX H
        assert_eq!(
            controls.len(),
            targets.len(),
            "Controls and targets must have same length"
        );
        for (&c, &t) in controls.iter().zip(targets.iter()) {
            self.add_h(&[t]);
            self.add_cx(&[c], &[t]);
            self.add_h(&[t]);
        }
        self
    }

    /// Check how many messages have been added
    #[must_use]
    pub fn message_count(&self) -> u32 {
        self.msg_count
    }

    /// Clear the builder and start fresh
    ///
    /// This method completely replaces the builder with a new instance,
    /// releasing any allocated memory. Use this when memory usage is a concern
    /// or when you want absolute certainty of a fresh state.
    ///
    /// For performance-critical code or when creating many messages in sequence,
    /// consider using `reset()` instead, which preserves memory allocation.
    ///
    /// After clearing, you'll need to configure the builder for the desired message type
    /// by calling `for_quantum_operations()` or `for_outcomes()`.
    pub fn clear(&mut self) -> &mut Self {
        *self = Self::new();
        self
    }

    /// Reset the builder state while preserving allocated memory
    ///
    /// Unlike `clear()`, this method preserves the allocated memory buffer
    /// for better performance when reusing the same builder multiple times.
    /// This is the recommended method for performance-critical code,
    /// especially when creating many messages in sequence.
    ///
    /// After resetting, you'll need to configure the builder for the desired message type
    /// by calling `for_quantum_operations()` or `for_outcomes()`:
    ///
    /// ```
    /// # use pecos_engines::byte_message::ByteMessageBuilder;
    /// let mut builder = ByteMessageBuilder::new();
    ///
    /// // Create first message
    /// let _ = builder.for_quantum_operations();
    /// builder.add_h(&[0]);
    /// let message1 = builder.build();
    ///
    /// // Reset and configure for next message
    /// builder.reset();
    /// let _ = builder.for_quantum_operations();
    /// builder.add_h(&[1]);
    /// let message2 = builder.build();
    /// ```
    ///
    /// If memory usage is a concern or you want to ensure a completely fresh state,
    /// consider using `clear()` instead.
    pub fn reset(&mut self) -> &mut Self {
        // Truncate the buffer to just the batch header size
        self.buffer.truncate(size_of::<BatchHeader>());

        // Zero out the batch header area more efficiently
        // Using slice fill is more efficient than a loop for small fixed-size areas
        self.buffer.fill(0);

        // Reset message count and mode
        self.msg_count = 0;
        self.mode = BuilderMode::Empty;

        self
    }

    /// Build the final message batch without type checking
    ///
    /// This creates a message without validating the builder's state, which is useful
    /// for internal usage or when you're confident the message is correctly constructed.
    ///
    /// # Returns
    ///
    /// Returns a `ByteMessage` containing the constructed binary message.
    #[must_use]
    pub fn build_unchecked(&mut self) -> ByteMessage {
        // Calculate total size and update batch header
        let total_size = self.buffer.len();

        // Create a batch header with proper message count and size
        let header = BatchHeader::new(
            self.msg_count,
            u32::try_from(total_size).unwrap_or_else(|_| {
                log::warn!("Message size exceeds u32::MAX, using maximum value");
                u32::MAX
            }),
        );

        // Write header to the start of the buffer
        self.buffer[0..size_of::<BatchHeader>()].copy_from_slice(bytes_of(&header));

        // Return a ByteMessage with the buffer
        ByteMessage::new(&self.buffer)
    }

    /// Build the message and return it
    ///
    /// Validates the builder state and constructs a final `ByteMessage` containing
    /// all added operations or outcomes.
    ///
    /// # Returns
    ///
    /// Returns a `ByteMessage` containing the constructed binary message.
    ///
    /// # Panics
    ///
    /// This function will panic if:
    /// - Messages have been added but the builder mode was not explicitly set
    ///   (call `for_quantum_operations()` or `for_outcomes()` before adding operations)
    #[must_use]
    pub fn build(&mut self) -> ByteMessage {
        // Validate that a mode was explicitly set if operations were added
        assert!(
            !(self.msg_count > 0 && self.mode == BuilderMode::Empty),
            "Builder mode not specified. Call for_quantum_operations() or for_outcomes() before adding operations."
        );

        // Complete the message by building the batch header
        self.build_unchecked()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::byte_message::GateType;
    use crate::byte_message::protocol::{BATCH_MAGIC, PROTOCOL_VERSION};
    use pecos_core::QubitId;

    #[test]
    fn test_gate_command_interface() {
        // Create a builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Add gates using new GateCommand interface
        let gate = Gate::h(&[0]);
        builder.add_gate_command(&gate);
        let gate = Gate::rz(Angle64::from_radians(0.5), &[1]);
        builder.add_gate_command(&gate);

        // Test multiple gates at once
        let gates = vec![Gate::x(&[2]), Gate::cx(&[(0, 1)])];
        builder.add_gate_commands(&gates);

        // Build and verify basic structure
        let message = builder.build();
        assert!(!message.is_empty().unwrap());
    }

    #[test]
    fn test_builder_basic() {
        // Create a builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Add some gates
        builder.add_h(&[0]);
        builder.add_cx(&[0], &[1]);
        builder.add_measurements(&[2]);

        // Build the message
        let message = builder.build();

        // Parse the message
        let commands = message.quantum_ops().unwrap();

        // Verify the commands
        assert_eq!(commands.len(), 3);
        assert_eq!(commands[0].gate_type, GateType::H);
        assert_eq!(commands[0].qubits.as_slice(), &[QubitId(0)]);
        assert_eq!(commands[1].gate_type, GateType::CX);
        assert_eq!(commands[1].qubits.as_slice(), &[QubitId(0), QubitId(1)]);
        assert_eq!(commands[2].gate_type, GateType::Measure);
        assert_eq!(commands[2].qubits.as_slice(), &[QubitId(2)]);
    }

    #[test]
    fn test_builder_measurement_message() {
        // Create a builder for measurement outcomes
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();

        // Add some measurement outcomes
        builder.add_outcomes(&[0]);

        // Build the message
        let message = builder.build();

        // No need to verify a specific message type anymore, just ensure it's valid
        assert!(message.is_empty().is_ok());
    }

    #[test]
    fn test_builder_gates() {
        // Create a builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Add various gates
        builder.add_h(&[0]);
        builder.add_x(&[1]);
        builder.add_y(&[2]);
        builder.add_z(&[3]);
        builder.add_rz(Angle64::from_radians(0.5), &[4]);
        builder.add_r1xy(Angle64::from_radians(0.1), Angle64::from_radians(0.2), &[5]);
        builder.add_measurements(&[6]);

        // Build the message
        let message = builder.build();

        // Parse the message
        let commands = message.quantum_ops().unwrap();

        // Verify the commands
        assert_eq!(commands.len(), 7);
        assert_eq!(commands[0].gate_type, GateType::H);
        assert_eq!(commands[1].gate_type, GateType::X);
        assert_eq!(commands[2].gate_type, GateType::Y);
        assert_eq!(commands[3].gate_type, GateType::Z);
        assert_eq!(commands[4].gate_type, GateType::RZ);
        // RZ angle is now stored in angles field (as Angle64), params should be empty
        assert_eq!(commands[4].angles.len(), 1);
        assert!((commands[4].angles[0].to_radians() - 0.5).abs() < 1e-10);
        assert!(commands[4].params.is_empty());
        assert_eq!(commands[5].gate_type, GateType::R1XY);
        // R1XY has two angles, also stored in angles field
        assert_eq!(commands[5].angles.len(), 2);
        assert!((commands[5].angles[0].to_radians() - 0.1).abs() < 1e-10);
        assert!((commands[5].angles[1].to_radians() - 0.2).abs() < 1e-10);
        assert!(commands[5].params.is_empty());
        assert_eq!(commands[6].gate_type, GateType::Measure);
    }

    #[test]
    #[should_panic(
        expected = "Cannot mix quantum operations and measurement outcomes in the same message"
    )]
    fn test_builder_type_checking() {
        // Create a builder for measurement outcomes
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_outcomes();

        // Try to add a gate (should panic)
        builder.add_h(&[0]);
    }

    #[test]
    fn test_builder_empty() {
        // Create an empty builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Build the message
        let message = builder.build();

        // Verify the message is empty
        assert!(message.is_empty().unwrap());
    }

    #[test]
    fn test_add_measure_collections() {
        // Create a builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Add measurements for multiple qubits
        let qubits = vec![0, 1, 2];
        builder.add_measurements(&qubits);

        // Build the message
        let message = builder.build();

        // Parse the message
        let commands = message.quantum_ops().unwrap();

        // Verify the commands
        assert_eq!(commands.len(), 3);
        for i in 0..3 {
            assert_eq!(commands[i].gate_type, GateType::Measure);
            assert_eq!(commands[i].qubits.as_slice(), &[QubitId(qubits[i])]);
        }
    }

    #[test]
    fn test_add_measure_leakages() {
        // Create a builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Add measure_leakages for multiple qubits
        let qubits = vec![0, 1, 2];
        builder.add_measure_leakages(&qubits);

        // Build the message
        let message = builder.build();

        // Parse the message
        let commands = message.quantum_ops().unwrap();

        // Verify the commands
        assert_eq!(commands.len(), 3);
        for i in 0..3 {
            assert_eq!(commands[i].gate_type, GateType::MeasureLeaked);
            assert_eq!(commands[i].qubits.as_slice(), &[QubitId(qubits[i])]);
        }
    }

    #[test]
    fn test_batch_structure() {
        // Create a builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Add a gate
        builder.add_h(&[0]);

        // Build the message
        let message = builder.build();

        // Verify the batch structure
        let bytes = message.as_bytes();
        assert!(bytes.len() >= size_of::<BatchHeader>());

        // Parse the batch header - guaranteed aligned at offset 0
        let batch_header =
            *bytemuck::from_bytes::<BatchHeader>(&bytes[0..size_of::<BatchHeader>()]);
        assert_eq!(batch_header.magic, BATCH_MAGIC);
        assert_eq!(batch_header.version, PROTOCOL_VERSION);
        assert_eq!(batch_header.msg_count, 1);
    }

    #[test]
    fn test_for_quantum_operations() {
        // Create a builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Add a gate
        builder.add_h(&[0]);

        // Build the message
        let message = builder.build();

        // Parse the message
        let commands = message.quantum_ops().unwrap();

        // Verify the commands
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].gate_type, GateType::H);
    }

    #[test]
    fn test_message_count_and_clear() {
        // Create a builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Add some gates
        builder.add_h(&[0]);
        builder.add_cx(&[0], &[1]);

        // Check the message count
        assert_eq!(builder.message_count(), 2);

        // Clear the builder
        builder.clear();

        // Check the message count after clearing
        assert_eq!(builder.message_count(), 0);

        // Add a new gate
        builder.add_h(&[0]);

        // Check the message count again
        assert_eq!(builder.message_count(), 1);
    }

    #[test]
    fn test_reset() {
        // Create a builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Add some gates
        builder.add_h(&[0]);
        builder.add_cx(&[0], &[1]);

        // Check the message count
        assert_eq!(builder.message_count(), 2);

        // Get the buffer capacity before reset
        let capacity_before = builder.buffer.capacity();

        // Reset the builder
        builder.reset();

        // Check the message count after reset
        assert_eq!(builder.message_count(), 0);

        // Verify the buffer capacity is preserved
        assert_eq!(builder.buffer.capacity(), capacity_before);

        // Configure for quantum operations again
        let _ = builder.for_quantum_operations();

        // Add a new gate
        builder.add_h(&[0]);

        // Check the message count again
        assert_eq!(builder.message_count(), 1);

        // Build the message and verify it's valid
        let message = builder.build();
        let commands = message.quantum_ops().unwrap();
        assert_eq!(commands.len(), 1);
        assert_eq!(commands[0].gate_type, GateType::H);
    }

    #[test]
    fn compare_clear_vs_reset_performance() {
        const ITERATIONS: usize = 5000;
        const TRIALS: usize = 5;

        let mut clear_durations = Vec::with_capacity(TRIALS);
        let mut reset_durations = Vec::with_capacity(TRIALS);

        for _ in 0..TRIALS {
            // Test with clear()
            let start_clear = std::time::Instant::now();
            {
                let mut builder = ByteMessageBuilder::new();

                for i in 0..ITERATIONS {
                    if i > 0 {
                        builder.clear();
                    }

                    // Configure for quantum operations
                    let _ = builder.for_quantum_operations();

                    // Add a gate
                    builder.add_h(&[0]);

                    // Build the message
                    let _message = builder.build();
                }
            }
            clear_durations.push(start_clear.elapsed());

            // Test with reset()
            let start_reset = std::time::Instant::now();
            {
                let mut builder = ByteMessageBuilder::new();

                for i in 0..ITERATIONS {
                    if i > 0 {
                        builder.reset();
                    }

                    // Configure for quantum operations
                    let _ = builder.for_quantum_operations();

                    // Add a gate
                    builder.add_h(&[0]);

                    // Build the message
                    let _message = builder.build();
                }
            }
            reset_durations.push(start_reset.elapsed());
        }

        // Calculate averages
        #[allow(clippy::cast_precision_loss)]
        let avg_clear = clear_durations
            .iter()
            .map(std::time::Duration::as_secs_f64)
            .sum::<f64>()
            / (TRIALS as f64);
        #[allow(clippy::cast_precision_loss)]
        let avg_reset = reset_durations
            .iter()
            .map(std::time::Duration::as_secs_f64)
            .sum::<f64>()
            / (TRIALS as f64);

        // Print results
        println!("Performance comparison ({TRIALS} trials of {ITERATIONS} iterations each):");
        println!("  clear() + for_quantum_operations(): {avg_clear:.6}s (average)");
        println!("  reset() + for_quantum_operations(): {avg_reset:.6}s (average)");
        println!("  reset() approach is {:.2}x faster", avg_clear / avg_reset);

        // We don't assert anything here as performance can vary by environment,
        // but reset() should generally be faster
    }
}
