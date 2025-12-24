use crate::byte_message::builder::ByteMessageBuilder;
use crate::byte_message::protocol::{
    BatchHeader, GateHeader, MessageHeader, MessageType, OutcomeHeader, ReturnValueHeader,
    calc_padding,
};
use log::trace;
use pecos_core::QubitId;
use pecos_core::errors::PecosError;
use pecos_core::gate_type::GateType;
use pecos_core::gates::Gate;
use std::mem::size_of;

/// A message encoded using the PECOS byte protocol
///
/// Uses Vec<u32> for guaranteed 4-byte alignment matching our protocol design
#[derive(Clone)]
pub struct ByteMessage {
    data: Vec<u32>,
    byte_len: usize,
}

impl ByteMessage {
    /// Create a new `ByteMessage` from raw bytes
    #[must_use]
    pub fn new(bytes: &[u8]) -> Self {
        let byte_len = bytes.len();

        if byte_len == 0 {
            return Self {
                data: Vec::new(),
                byte_len: 0,
            };
        }

        // Calculate word count (round up to 4-byte boundary)
        let word_count = byte_len.div_ceil(4);

        // Create aligned storage
        let mut data = vec![0u32; word_count];

        // Copy bytes into aligned storage
        let data_bytes = bytemuck::cast_slice_mut::<u32, u8>(&mut data);
        data_bytes[..byte_len].copy_from_slice(bytes);

        Self { data, byte_len }
    }

    /// Create a new message builder
    #[must_use]
    pub fn builder() -> ByteMessageBuilder {
        ByteMessageBuilder::new()
    }

    /// Create a new `ByteMessage` from already-aligned u32 data
    ///
    /// This method is used when receiving data from FFI boundaries where
    /// the data is already guaranteed to be 4-byte aligned.
    #[must_use]
    pub fn from_aligned_u32_data(data: Vec<u32>, byte_len: usize) -> Self {
        Self { data, byte_len }
    }

    /// Get a reference to the raw bytes
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        if self.byte_len == 0 {
            return &[];
        }

        let all_bytes = bytemuck::cast_slice::<u32, u8>(&self.data);
        &all_bytes[..self.byte_len]
    }

    /// Consume the message and return the raw bytes
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        if self.byte_len == 0 {
            return Vec::new();
        }

        let all_bytes = bytemuck::cast_slice::<u32, u8>(&self.data);
        all_bytes[..self.byte_len].to_vec()
    }

    /// Create a new message builder pre-configured for quantum operations
    ///
    /// This is a convenience method that creates a new builder and configures it
    /// for quantum operations.
    ///
    /// # Returns
    ///
    /// A `MessageBuilder` configured for quantum operations.
    #[must_use]
    pub fn quantum_operations_builder() -> ByteMessageBuilder {
        let mut builder = Self::builder();
        let _ = builder.for_quantum_operations();
        builder
    }

    /// Create a new message builder pre-configured for measurement outcomes
    ///
    /// This is a convenience method that creates a new builder and configures it
    /// for measurement outcomes.
    ///
    /// # Returns
    ///
    /// A `MessageBuilder` configured for measurement outcomes.
    #[must_use]
    pub fn outcomes_builder() -> ByteMessageBuilder {
        let mut builder = Self::builder();
        let _ = builder.for_outcomes();
        builder
    }

    /// Create a new empty message
    ///
    /// This is a convenience method that creates a new empty message.
    /// Empty messages are used when no quantum operations are needed.
    ///
    /// # Returns
    ///
    /// A `ByteMessage` containing an empty batch.
    #[must_use]
    pub fn create_empty() -> Self {
        let mut builder = ByteMessageBuilder::new();
        builder.build()
    }

    /// Determine the message type by parsing the header
    ///
    /// This function parses the message header to determine the type of the message.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the `MessageType` if successful, or a `PecosError` if there was an error.
    ///
    /// # Errors
    ///
    /// This function may return a `PecosError::InvalidInput` if:
    /// - The message is too small to contain a batch header
    /// - The batch header is invalid
    /// - The batch contains no messages
    /// - The message is too small to contain a message header
    /// - The message header contains an invalid message type
    pub fn message_type(&self) -> Result<MessageType, PecosError> {
        // Parse and validate the batch header
        let batch_header = self.parse_batch_header()?;

        // Need at least one message to determine type
        if batch_header.msg_count == 0 {
            return Err(PecosError::Input("Batch contains no messages".to_string()));
        }

        // Parse the first message header
        let (msg_header, _) = self.parse_message_header(size_of::<BatchHeader>())?;

        msg_header
            .get_type()
            .map_err(|e| PecosError::Input(format!("Failed to determine message type: {e}")))
    }

    // Private helper methods

    // Helper function to check if the message has no data.
    // Returns true if either the byte length is 0 or the data vector is empty.
    fn has_no_data(&self) -> bool {
        self.byte_len == 0 || self.data.is_empty()
    }

    /// Parse and validate the batch header
    fn parse_batch_header(&self) -> Result<BatchHeader, PecosError> {
        if self.byte_len < size_of::<BatchHeader>() {
            return Err(PecosError::Input(
                "Message too small for batch header".to_string(),
            ));
        }

        // Parse batch header - guaranteed aligned at offset 0 due to Vec<u32> storage
        let batch_header =
            *bytemuck::from_bytes::<BatchHeader>(&self.as_bytes()[0..size_of::<BatchHeader>()]);

        if !batch_header.is_valid() {
            return Err(PecosError::Input("Invalid batch header".to_string()));
        }

        Ok(batch_header)
    }

    /// Parse a message header at the given offset
    fn parse_message_header(&self, offset: usize) -> Result<(MessageHeader, usize), PecosError> {
        if offset + size_of::<MessageHeader>() > self.byte_len {
            return Err(PecosError::Input(
                "Message too small for message header".to_string(),
            ));
        }

        // Parse message header - guaranteed aligned due to builder padding
        let msg_header = *bytemuck::from_bytes::<MessageHeader>(
            &self.as_bytes()[offset..offset + size_of::<MessageHeader>()],
        );

        // Return the header and the new offset after the header
        Ok((msg_header, offset + size_of::<MessageHeader>()))
    }

    /// Process a single message from the buffer, returning a gate
    ///
    /// This is a helper method used by `quantum_ops` to process gate messages.
    ///
    /// # Arguments
    ///
    /// * `offset` - The offset in the buffer to start processing from
    ///
    /// # Returns
    ///
    /// Returns a tuple of:
    /// - The new offset after processing this message
    /// - An Option containing a Gate operation if one was found
    ///
    /// # Errors
    ///
    /// Returns an error if the message is malformed.
    fn process_gate_message(&self, offset: usize) -> Result<(usize, Option<Gate>), PecosError> {
        // Parse message header
        let Ok((msg_header, new_offset)) = self.parse_message_header(offset) else {
            // If we can't parse the header, just return the current offset with no gate
            return Ok((offset, None));
        };
        let offset = new_offset;

        // Get message type
        let Ok(msg_type) = msg_header.get_type() else {
            // Skip invalid message types
            trace!("Skipping message with invalid type");

            // Calculate the new offset after this message
            let payload_size = msg_header.payload_size as usize;
            let payload_end = offset + payload_size;
            let padding = calc_padding(payload_size, 4);
            let new_offset = payload_end + (if padding > 0 { padding } else { 0 });

            return Ok((new_offset, None));
        };

        // Check payload bounds
        let payload_size = msg_header.payload_size as usize;
        let payload_end = offset + payload_size;

        // Make sure the payload fits within the buffer
        if payload_end > self.byte_len {
            return Err(PecosError::Input(format!(
                "Message payload extends beyond message bounds: offset={}, size={}, total_len={}",
                offset, payload_size, self.byte_len
            )));
        }

        // Extract the payload
        let payload = &self.as_bytes()[offset..payload_end];

        // Process based on message type - we only care about Gate messages here
        let result = if msg_type == MessageType::Gate {
            // Debug: dump payload bytes for RZ gates
            if payload.len() >= size_of::<GateHeader>() {
                let header =
                    *bytemuck::from_bytes::<GateHeader>(&payload[0..size_of::<GateHeader>()]);
                if header.gate_type == GateType::RZ as u8 {
                    trace!("process_gate_message: RZ gate payload dump:");
                    trace!("  Total payload size: {} bytes", payload.len());
                    trace!(
                        "  Header: gate_type={}, num_qubits={}, has_params={}",
                        header.gate_type, header.num_qubits, header.has_params
                    );

                    // Dump raw bytes in hex
                    let hex_bytes: Vec<String> =
                        payload.iter().map(|b| format!("{b:02x}")).collect();
                    trace!("  Raw bytes: {}", hex_bytes.join(" "));
                }
            }

            match Self::parse_gate_command(payload) {
                Ok(cmd) => Some(cmd),
                Err(e) => {
                    trace!("Error parsing gate: {e}");
                    None
                }
            }
        } else {
            None
        };

        // Calculate the new offset after this message
        let padding = calc_padding(payload_size, 4);
        let new_offset = payload_end + (if padding > 0 { padding } else { 0 });

        Ok((new_offset, result))
    }

    /// Process a single message from the buffer, returning an outcome value
    ///
    /// This is a helper method used by outcomes to process outcome messages.
    ///
    /// # Arguments
    ///
    /// * `offset` - The offset in the buffer to start processing from
    ///
    /// # Returns
    ///
    /// Returns a tuple of:
    /// - The new offset after processing this message
    /// - An Option containing a measurement outcome if one was found
    ///
    /// # Errors
    ///
    /// Returns an error if the message is malformed.
    fn process_outcome_message(&self, offset: usize) -> Result<(usize, Option<u32>), PecosError> {
        // Parse message header
        let Ok((msg_header, new_offset)) = self.parse_message_header(offset) else {
            // If we can't parse the header, just return the current offset with no outcome
            return Ok((offset, None));
        };
        let offset = new_offset;

        // Get message type
        let Ok(msg_type) = msg_header.get_type() else {
            // Skip invalid message types
            trace!("Skipping message with invalid type");

            // Calculate the new offset after this message
            let payload_size = msg_header.payload_size as usize;
            let payload_end = offset + payload_size;
            let padding = calc_padding(payload_size, 4);
            let new_offset = payload_end + (if padding > 0 { padding } else { 0 });

            return Ok((new_offset, None));
        };

        // Check payload bounds
        let payload_size = msg_header.payload_size as usize;
        let payload_end = offset + payload_size;

        // Make sure the payload fits within the buffer
        if payload_end > self.byte_len {
            return Err(PecosError::Input(format!(
                "Message payload extends beyond message bounds: offset={}, size={}, total_len={}",
                offset, payload_size, self.byte_len
            )));
        }

        // Extract the payload
        let payload = &self.as_bytes()[offset..payload_end];

        // Process based on message type - we only care about Outcome messages here
        let result = if msg_type == MessageType::Outcome {
            if payload.len() >= size_of::<OutcomeHeader>() {
                // OutcomeHeader at aligned payload start
                let result_header =
                    *bytemuck::from_bytes::<OutcomeHeader>(&payload[0..size_of::<OutcomeHeader>()]);
                Some(result_header.outcome)
            } else {
                None
            }
        } else {
            None
        };

        // Calculate the new offset after this message
        let padding = calc_padding(payload_size, 4);
        let new_offset = payload_end + (if padding > 0 { padding } else { 0 });

        Ok((new_offset, result))
    }

    /// Check if this message is empty (contains no operations).
    ///
    /// # Returns
    ///
    /// Returns `Ok(true)` if the message is empty, `Ok(false)` if it contains operations.
    ///
    /// # Errors
    ///
    /// Returns a `PecosError` if there was an error parsing the message structure.
    pub fn is_empty(&self) -> Result<bool, PecosError> {
        // First check if this is an empty message with no data
        if self.has_no_data() {
            return Ok(true);
        }

        // Parse and validate the batch header
        let batch_header = self.parse_batch_header()?;

        // Message is empty if it has no messages
        if batch_header.msg_count == 0 {
            return Ok(true);
        }

        // Otherwise, check if there are any actual operations
        let commands = self.quantum_ops()?;
        Ok(commands.is_empty())
    }

    /// Parse quantum operations from this message
    ///
    /// # Errors
    ///
    /// Returns an error if the message is malformed or contains invalid quantum operations.
    pub fn quantum_ops(&self) -> Result<Vec<Gate>, PecosError> {
        // Parse and validate the batch header
        let batch_header = self.parse_batch_header()?;

        trace!(
            "quantum_ops: Processing {} messages",
            batch_header.msg_count
        );

        let mut commands = Vec::new();
        let mut offset = size_of::<BatchHeader>();

        // Process each message
        for msg_idx in 0..batch_header.msg_count {
            // Try to process this message
            let (new_offset, maybe_gate) = self.process_gate_message(offset)?;
            offset = new_offset;

            // Add any gate we found to our commands list
            if let Some(gate) = maybe_gate {
                trace!("quantum_ops: Message {msg_idx} parsed as gate: {gate:?}");
                commands.push(gate);
            } else {
                trace!("quantum_ops: Message {msg_idx} did not yield a gate");
            }
        }

        trace!("quantum_ops: Total gates parsed: {}", commands.len());

        Ok(commands)
    }

    /// Parse measurement outcomes from this message
    ///
    /// # Errors
    ///
    /// Returns an error if the message is malformed or contains invalid outcome data.
    pub fn outcomes(&self) -> Result<Vec<u32>, PecosError> {
        // Parse and validate the batch header
        let batch_header = self.parse_batch_header()?;

        let mut measurements = Vec::new();
        let mut offset = size_of::<BatchHeader>();

        // Process each message
        for _ in 0..batch_header.msg_count {
            // Try to process this message directly for outcomes
            let (new_offset, maybe_outcome) = self.process_outcome_message(offset)?;
            offset = new_offset;

            // Add any outcome we found to our measurements list
            if let Some(outcome) = maybe_outcome {
                measurements.push(outcome);
            }
        }

        Ok(measurements)
    }

    /// Extract return value from the message.
    ///
    /// # Returns
    ///
    /// Returns the return value if found, or None if no return value is present.
    ///
    /// # Errors
    ///
    /// Returns an error if the message is malformed.
    pub fn return_value(&self) -> Result<Option<i64>, PecosError> {
        // Parse and validate the batch header
        let batch_header = self.parse_batch_header()?;

        let mut offset = size_of::<BatchHeader>();

        // Process each message
        for _ in 0..batch_header.msg_count {
            // Try to process this message for return value
            let (new_offset, maybe_value) = self.process_return_value_message(offset)?;
            offset = new_offset;

            // If we found a return value, return it immediately
            if let Some(value) = maybe_value {
                return Ok(Some(value));
            }
        }

        Ok(None)
    }

    /// Process a single message to extract return value if it's a `ReturnValue` message
    fn process_return_value_message(
        &self,
        offset: usize,
    ) -> Result<(usize, Option<i64>), PecosError> {
        // Parse message header
        let Ok((msg_header, new_offset)) = self.parse_message_header(offset) else {
            // If we can't parse the header, just return the current offset with no value
            return Ok((offset, None));
        };
        let offset = new_offset;

        // Get message type
        let Ok(msg_type) = msg_header.get_type() else {
            // Skip invalid message types
            trace!("Skipping message with invalid type");

            // Calculate the new offset after this message
            let payload_size = msg_header.payload_size as usize;
            let payload_end = offset + payload_size;
            let padding = calc_padding(payload_size, 4);
            let new_offset = payload_end + (if padding > 0 { padding } else { 0 });

            return Ok((new_offset, None));
        };

        // Check payload bounds
        let payload_size = msg_header.payload_size as usize;
        let payload_end = offset + payload_size;

        // Make sure the payload fits within the buffer
        if payload_end > self.byte_len {
            return Err(PecosError::Input(format!(
                "Message payload extends beyond message bounds: offset={}, size={}, total_len={}",
                offset, payload_size, self.byte_len
            )));
        }

        // Extract the payload
        let payload = &self.as_bytes()[offset..payload_end];

        // Process based on message type - we only care about ReturnValue messages here
        let result = if msg_type == MessageType::ReturnValue {
            if payload.len() >= size_of::<ReturnValueHeader>() {
                // ReturnValueHeader at aligned payload start
                let return_header = *bytemuck::from_bytes::<ReturnValueHeader>(
                    &payload[0..size_of::<ReturnValueHeader>()],
                );
                Some(return_header.value)
            } else {
                None
            }
        } else {
            None
        };

        // Calculate the new offset after this message
        let padding = calc_padding(payload_size, 4);
        let new_offset = payload_end + (if padding > 0 { padding } else { 0 });

        Ok((new_offset, result))
    }

    /// Validate if the payload has enough bytes for the gate header
    fn validate_gate_payload_size(payload: &[u8]) -> Result<(), PecosError> {
        if payload.len() < size_of::<GateHeader>() {
            return Err(PecosError::Input(
                "Quantum gate message payload too small".to_string(),
            ));
        }
        Ok(())
    }

    /// Validate if the payload has enough bytes for qubit indices
    fn validate_qubit_indices_size(
        payload: &[u8],
        qubits_offset: usize,
        qubits_size: usize,
    ) -> Result<(), PecosError> {
        let minimum_size = qubits_offset + qubits_size;
        if payload.len() < minimum_size {
            return Err(PecosError::Input(
                "Quantum gate message payload too small for qubit indices".to_string(),
            ));
        }
        Ok(())
    }

    /// Parse qubit indices from the payload and convert to `QubitIds` directly
    fn parse_qubit_indices(
        payload: &[u8],
        qubits_offset: usize,
        num_qubits: usize,
    ) -> Vec<QubitId> {
        let mut qubits = Vec::with_capacity(num_qubits);
        for i in 0..num_qubits {
            let qubit_offset = qubits_offset + i * size_of::<u32>();
            let qubit = u32::from_le_bytes([
                payload[qubit_offset],
                payload[qubit_offset + 1],
                payload[qubit_offset + 2],
                payload[qubit_offset + 3],
            ]) as usize;
            qubits.push(QubitId::from(qubit));
        }
        qubits
    }

    /// Parse gate parameters based on gate type
    fn parse_gate_parameters(
        payload: &[u8],
        params_offset: usize,
        gate_type: GateType,
    ) -> Result<Vec<f64>, PecosError> {
        // Get the number of parameters this gate type requires
        let param_count = gate_type.classical_arity();
        if param_count == 0 {
            return Ok(Vec::new());
        }

        trace!("parse_gate_parameters: Gate {gate_type:?} requires {param_count} parameters");

        // Validate the parameter size
        let required_size = param_count * size_of::<f64>();
        Self::validate_params_size(
            payload,
            params_offset,
            required_size,
            &format!("{gate_type:?} parameters"),
        )?;

        // Parse all parameters
        let mut params = Vec::with_capacity(param_count);
        for i in 0..param_count {
            let param_offset = params_offset + i * size_of::<f64>();
            let param = Self::parse_f64_param(payload, param_offset);
            trace!("parse_gate_parameters: Parameter {i} at offset {param_offset}: {param}");
            params.push(param);
        }

        // Special logging for RZ gate parameters
        if matches!(gate_type, GateType::RZ) && !params.is_empty() {
            trace!(
                "parse_gate_parameters: RZ angle parsed as {} radians ({} degrees)",
                params[0],
                params[0].to_degrees()
            );
        }

        Ok(params)
    }

    /// Validate if the payload has enough bytes for parameters
    fn validate_params_size(
        payload: &[u8],
        params_offset: usize,
        required_size: usize,
        gate_name: &str,
    ) -> Result<(), PecosError> {
        if payload.len() < params_offset + required_size {
            return Err(PecosError::Input(format!(
                "Quantum gate message payload too small for {gate_name}"
            )));
        }
        Ok(())
    }

    /// Parse an f64 parameter from the payload
    fn parse_f64_param(payload: &[u8], offset: usize) -> f64 {
        let param_bytes = &payload[offset..offset + size_of::<f64>()];
        // Performance critical path during simulation - slice to array conversion should never fail
        // when we already verified the buffer size (8 bytes for f64)
        f64::from_le_bytes(
            param_bytes[..8]
                .try_into()
                .expect("Byte buffer has incorrect length for f64 conversion"),
        )
    }

    /// Parse a quantum gate message payload to `Gate`
    fn parse_gate_command(payload: &[u8]) -> Result<Gate, PecosError> {
        Self::validate_gate_payload_size(payload)?;

        // Parse gate header - guaranteed aligned since payload starts at aligned boundary
        let header = *bytemuck::from_bytes::<GateHeader>(&payload[0..size_of::<GateHeader>()]);
        let num_qubits = header.num_qubits as usize;
        let has_params = header.has_params != 0;
        let gate_type = GateType::from(header.gate_type);

        trace!(
            "parse_gate_command: Parsing gate type {gate_type:?}, num_qubits: {num_qubits}, has_params: {has_params}"
        );

        // Calculate sizes
        let qubits_byte_size = num_qubits * size_of::<u32>();
        let qubits_offset = size_of::<GateHeader>();

        Self::validate_qubit_indices_size(payload, qubits_offset, qubits_byte_size)?;

        // Parse qubit indices directly to QubitId
        let qubits = Self::parse_qubit_indices(payload, qubits_offset, num_qubits);

        trace!("parse_gate_command: Parsed qubits: {qubits:?}");

        // Parse parameters if present
        let params = if has_params {
            let params_offset = qubits_offset + qubits_byte_size;
            let parsed_params = Self::parse_gate_parameters(payload, params_offset, gate_type)?;
            trace!("parse_gate_command: Parsed parameters: {parsed_params:?}");
            parsed_params
        } else {
            Vec::new()
        };

        // Special logging for RZ gates
        if matches!(gate_type, GateType::RZ) {
            trace!(
                "parse_gate_command: RZ gate parsed with angle: {:?}, qubit: {:?}",
                params.first(),
                qubits.first()
            );
        }

        Ok(Gate::new(gate_type, params, qubits))
    }

    // The parse_simple_measurement method has been removed as part of simplifying the protocol.
    // All measurements are now handled as regular gates through parse_gate_command.
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;
    use crate::quantum::StateVecEngine;
    use pecos_core::QubitId;

    #[test]
    fn test_bytemap_builder() {
        // Create a message with H and CX gates
        let mut builder = ByteMessage::quantum_operations_builder();
        builder.add_h(&[0]);
        builder.add_cx(&[0], &[1]);
        let message = builder.build();

        // Parse the message
        let parsed_commands = message.quantum_ops().unwrap();
        assert_eq!(parsed_commands.len(), 2);
        assert_eq!(parsed_commands[0].gate_type, GateType::H);
        assert_eq!(parsed_commands[0].qubits, vec![QubitId(0)]);
        assert_eq!(parsed_commands[1].gate_type, GateType::CX);
        assert_eq!(parsed_commands[1].qubits, vec![QubitId(0), QubitId(1)]);
    }

    #[test]
    fn test_message_type() {
        // Create an empty message
        let empty_message = ByteMessage::create_empty();

        // Empty message should be parseable
        assert!(empty_message.is_empty().unwrap());

        // Create a quantum operations message
        let mut builder = ByteMessage::quantum_operations_builder();
        builder.add_h(&[0]);
        let quantum_message = builder.build();

        // Check that we can parse the gates
        let ops = quantum_message.quantum_ops().unwrap();
        assert_eq!(ops.len(), 1);

        // Create a measurement results message
        let mut builder = ByteMessage::outcomes_builder();
        builder.add_outcomes(&[0]);
        let results_message = builder.build();

        // Check that we can parse the outcomes
        let outcomes = results_message.outcomes().unwrap();
        assert_eq!(outcomes.len(), 1);
    }

    #[test]
    fn test_parse_measurements() {
        // Create a message with measurement results
        let mut builder = ByteMessage::outcomes_builder();
        builder.add_outcomes(&[0, 1]);
        let message = builder.build();

        // Parse the measurements
        let measurements = message.outcomes().unwrap();
        assert_eq!(measurements.len(), 2);

        // The measurements now just return outcomes
        assert_eq!(measurements[0], 0);
        assert_eq!(measurements[1], 1);
    }

    #[test]
    fn test_parse_measurements_with_indexing() {
        // Create a message with measurement results
        let mut builder = ByteMessage::outcomes_builder();
        builder.add_outcomes(&[0, 1, 0]);
        let message = builder.build();

        // Get the raw measurement results
        let outcomes = message.outcomes().unwrap();

        // Verify the outcomes match the input
        assert_eq!(outcomes.len(), 3);
        assert_eq!(outcomes[0], 0);
        assert_eq!(outcomes[1], 1);
        assert_eq!(outcomes[2], 0);

        // Convert raw outcomes to indexed results for easier assertions
        let results: Vec<(usize, u32)> = outcomes.into_iter().enumerate().collect();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], (0, 0));
        assert_eq!(results[1], (1, 1));
        assert_eq!(results[2], (2, 0));

        // Verify the types are correct
        let (result_id, outcome) = results[0];
        let _: usize = result_id;
        let _: u32 = outcome;
    }

    #[test]
    fn test_bell_state_measurements() {
        // Create a Bell state circuit: H on qubit 0, CX from 0 to 1, measure both qubits
        let mut builder = ByteMessage::quantum_operations_builder();

        // Apply H to qubit 0
        builder.add_h(&[0]);

        // Apply CX with control=0, target=1
        builder.add_cx(&[0], &[1]);

        // Measure qubit 0 with result_id 0
        builder.add_measurements(&[0]);

        // Measure qubit 1 with result_id 1
        builder.add_measurements(&[1]);

        let bell_circuit = builder.build();

        // Run the circuit multiple times and check the results
        let mut engine = StateVecEngine::new(2); // Create a simulator with 2 qubits

        for _ in 0..10 {
            // Reset the engine for each run
            engine.reset().unwrap();

            // Process the circuit
            let result_message = engine.process(bell_circuit.clone()).unwrap();

            // Get the raw measurement results
            let outcomes = result_message.outcomes().unwrap();

            // We know the measurement order: qubit 0 was measured first, then qubit 1
            assert_eq!(outcomes.len(), 2, "Expected exactly 2 measurement results");

            // The outcomes are now indexed by measurement order
            let q0_result = outcomes[0] != 0; // First measurement was qubit 0
            let q1_result = outcomes[1] != 0; // Second measurement was qubit 1

            // In a Bell state, the qubits should always have the same measurement outcome
            assert_eq!(
                q0_result, q1_result,
                "Qubits in Bell state should have correlated measurements"
            );
        }
    }

    #[test]
    fn test_is_empty() {
        // Create an empty message
        let empty_message = ByteMessage::create_empty();
        assert!(empty_message.is_empty().unwrap());

        // Create a non-empty message
        let non_empty_message = ByteMessage::quantum_operations_builder()
            .add_h(&[0])
            .build();
        assert!(!non_empty_message.is_empty().unwrap());
    }

    #[test]
    fn test_measurement_result_order_preservation() {
        // Test that measurement results maintain their order through ByteMessage
        let mut builder = ByteMessage::outcomes_builder();

        // Add measurement results in a specific order
        builder.add_outcomes(&[1]); // First result: 1
        builder.add_outcomes(&[0]); // Second result: 0
        builder.add_outcomes(&[1]); // Third result: 1
        builder.add_outcomes(&[1]); // Fourth result: 1
        builder.add_outcomes(&[0]); // Fifth result: 0

        let message = builder.build();

        // Parse the measurements back
        let results = message.outcomes().unwrap();

        // Verify order is preserved
        assert_eq!(results.len(), 5);
        assert_eq!(results[0], 1, "First result should be 1");
        assert_eq!(results[1], 0, "Second result should be 0");
        assert_eq!(results[2], 1, "Third result should be 1");
        assert_eq!(results[3], 1, "Fourth result should be 1");
        assert_eq!(results[4], 0, "Fifth result should be 0");

        // Also convert raw outcomes to indexed results
        let outcomes2 = message.outcomes().unwrap();
        let indexed_results: Vec<(usize, u32)> = outcomes2.into_iter().enumerate().collect();
        assert_eq!(indexed_results.len(), 5);
        assert_eq!(indexed_results[0], (0, 1), "First indexed result");
        assert_eq!(indexed_results[1], (1, 0), "Second indexed result");
        assert_eq!(indexed_results[2], (2, 1), "Third indexed result");
        assert_eq!(indexed_results[3], (3, 1), "Fourth indexed result");
        assert_eq!(indexed_results[4], (4, 0), "Fifth indexed result");
    }

    #[test]
    fn test_alignment_guarantees() {
        // Test various buffer sizes to ensure alignment is guaranteed
        for size in [0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 32, 1024] {
            let test_data: Vec<u8> = (0..size).map(|i| u8::try_from(i % 256).unwrap()).collect();
            let message = ByteMessage::new(&test_data);
            let bytes = message.as_bytes();

            // Verify data integrity
            assert_eq!(
                bytes,
                &test_data[..],
                "Data integrity check failed for size {size}"
            );

            // Verify alignment - the internal buffer should be 4-byte aligned
            // We can't directly test the internal alignment, but we can test that
            // our bytemuck calls work without fallback by creating structures
            if bytes.len() >= 4 {
                // Try to parse as u32 - guaranteed aligned at offset 0
                let _test_u32 = *bytemuck::from_bytes::<u32>(&bytes[0..4]);
                // If we reach here, parsing is working correctly
            }
        }
    }

    #[test]
    fn test_measurement_gate_order_preservation() {
        // Test that measurement gate order is preserved
        let mut builder = ByteMessage::quantum_operations_builder();

        // Add measurements of different qubits in specific order
        builder.add_measurements(&[3]); // First: measure qubit 3
        builder.add_measurements(&[1]); // Second: measure qubit 1
        builder.add_measurements(&[4]); // Third: measure qubit 4
        builder.add_measurements(&[0]); // Fourth: measure qubit 0
        builder.add_measurements(&[2]); // Fifth: measure qubit 2

        let message = builder.build();

        // Parse operations back
        let operations = message.quantum_ops().unwrap();

        // Verify we have 5 measurement operations in the correct order
        assert_eq!(operations.len(), 5);

        assert_eq!(operations[0].gate_type, GateType::Measure);
        assert_eq!(operations[0].qubits, vec![QubitId(3)]);

        assert_eq!(operations[1].gate_type, GateType::Measure);
        assert_eq!(operations[1].qubits, vec![QubitId(1)]);

        assert_eq!(operations[2].gate_type, GateType::Measure);
        assert_eq!(operations[2].qubits, vec![QubitId(4)]);

        assert_eq!(operations[3].gate_type, GateType::Measure);
        assert_eq!(operations[3].qubits, vec![QubitId(0)]);

        assert_eq!(operations[4].gate_type, GateType::Measure);
        assert_eq!(operations[4].qubits, vec![QubitId(2)]);
    }
}
