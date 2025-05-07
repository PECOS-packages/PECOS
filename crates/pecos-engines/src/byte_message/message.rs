use crate::byte_message::builder::ByteMessageBuilder;
use crate::byte_message::gate_type::{GateType, QuantumGate};
use crate::byte_message::protocol::{
    BatchHeader, MeasurementHeader, MeasurementResultHeader, MessageHeader, MessageType,
    QuantumGateHeader, calc_padding,
};
use crate::errors::QueueError;
use bytemuck::from_bytes;
use log::trace;
use std::mem::size_of;

/// A message encoded using the PECOS byte protocol
#[derive(Clone)]
pub struct ByteMessage {
    bytes: Vec<u8>,
}

impl ByteMessage {
    /// Create a new `ByteMessage` from raw bytes
    #[must_use]
    pub fn new(bytes: Vec<u8>) -> Self {
        Self { bytes }
    }

    /// Get a reference to the raw bytes
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Consume the message and return the raw bytes
    #[must_use]
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Create a new message builder
    #[must_use]
    pub fn builder() -> ByteMessageBuilder {
        ByteMessageBuilder::new()
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

    /// Create a new message builder pre-configured for measurement results
    ///
    /// This is a convenience method that creates a new builder and configures it
    /// for measurement results.
    ///
    /// # Returns
    ///
    /// A `MessageBuilder` configured for measurement results.
    #[must_use]
    pub fn measurement_results_builder() -> ByteMessageBuilder {
        let mut builder = Self::builder();
        let _ = builder.for_measurement_results();
        builder
    }

    /// Create a new flush message
    ///
    /// This is a convenience method that creates a new message with a flush command.
    /// Flush messages are used to signal the end of a batch of commands.
    ///
    /// # Returns
    ///
    /// A `ByteMessage` containing a flush command.
    #[must_use]
    pub fn create_flush() -> Self {
        let mut builder = ByteMessageBuilder::new();
        builder.add_flush(true);
        builder.build()
    }

    /// Create a new message with a circuit of quantum gates
    ///
    /// This is a convenience method that creates a new message with multiple quantum gates
    /// representing a quantum circuit.
    ///
    /// # Arguments
    ///
    /// * `gates` - A slice of `QuantumGate` objects
    ///
    /// # Returns
    ///
    /// A Result containing a `ByteMessage` with the circuit if successful, or a `QueueError` if there was an error.
    ///
    /// # Errors
    ///
    /// This function may return a `QueueError` if:
    /// - There is an error adding the gates to the builder
    /// - There is an error building the message
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_engines::byte_message::ByteMessage;
    /// use pecos_engines::byte_message::QuantumGate;
    ///
    /// // Create a circuit with H and CX gates
    /// let gates = vec![
    ///     QuantumGate::h(0),
    ///     QuantumGate::cx(0, 1),
    /// ];
    ///
    /// let message = ByteMessage::create_circuit_from_quantum_gates(&gates).unwrap();
    /// ```
    pub fn create_circuit_from_quantum_gates(gates: &[QuantumGate]) -> Result<Self, QueueError> {
        let mut builder = Self::quantum_operations_builder();
        builder.add_quantum_gates(gates);
        Ok(builder.build())
    }

    /// Create a new message from a sequence of command strings
    ///
    /// This is a convenience method that creates a new message from a sequence of command strings
    /// in the format "`GATE_TYPE` [params...] qubit1 qubit2 ...".
    ///
    /// # Arguments
    ///
    /// * `commands` - A slice of command strings to parse
    ///
    /// # Returns
    ///
    /// A Result containing a `ByteMessage` with the commands if successful, or a `QueueError` if there was an error.
    ///
    /// # Errors
    ///
    /// This function may return a `QueueError` if:
    /// - A command string has an invalid format
    /// - A command string contains an unknown gate type
    /// - A command string contains invalid parameters (e.g., non-numeric values for angles)
    /// - A command string contains invalid qubit indices
    pub fn create_from_commands(commands: &[&str]) -> Result<Self, QueueError> {
        let mut builder = Self::quantum_operations_builder();
        for cmd in commands {
            Self::parse_command_to_builder(&mut builder, cmd)?;
        }
        Ok(builder.build())
    }

    /// Record measurement results
    ///
    /// This is a convenience method that creates a new message with measurement results.
    /// It's used to report measurement outcomes back to the classical controller.
    ///
    /// # Arguments
    ///
    /// * `result_pairs` - A slice of tuples containing (`result_id`, outcome)
    ///   where `result_id` corresponds to the ID used when requesting the measurement
    ///   and outcome is the measurement result (typically 0 or 1)
    ///
    /// # Returns
    ///
    /// A `ByteMessage` containing the measurement results.
    #[must_use]
    pub fn record_measurement_results(result_pairs: &[(usize, u32)]) -> Self {
        let mut builder = Self::measurement_results_builder();

        // Collect result_ids and outcomes into separate vectors
        let mut result_ids = Vec::with_capacity(result_pairs.len());
        let mut outcomes = Vec::with_capacity(result_pairs.len());

        for (result_id, outcome) in result_pairs {
            result_ids.push(*result_id);
            outcomes.push(*outcome as usize); // Convert u32 to usize
        }

        builder.add_measurement_results(&outcomes, &result_ids);
        builder.build()
    }

    /// Create a message with a single quantum gate
    ///
    /// This is a convenience method that creates a new message with a single quantum gate.
    ///
    /// # Arguments
    ///
    /// * `gate` - The quantum gate to add
    ///
    /// # Returns
    ///
    /// A `ByteMessage` with the gate.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_engines::byte_message::ByteMessage;
    /// use pecos_engines::byte_message::QuantumGate;
    ///
    /// // Create a message with an H gate on qubit 0
    /// let gate = QuantumGate::h(0);
    /// let message = ByteMessage::create_with_quantum_gate(&gate);
    /// ```
    #[must_use]
    pub fn create_with_quantum_gate(gate: &QuantumGate) -> Self {
        let mut builder = Self::quantum_operations_builder();
        builder.add_quantum_gate(gate);
        builder.build()
    }

    /// Parse a command string and add it to the `ByteMessage` builder
    ///
    /// This function parses a command string in the format "`GATE_TYPE` [params...] qubit1 qubit2 ..."
    /// and adds the corresponding quantum gate to the provided builder.
    ///
    /// # Arguments
    ///
    /// * `builder` - The `ByteMessageBuilder` to add the command to
    /// * `cmd` - The command string to parse
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if the command was successfully parsed and added to the builder,
    /// or a `QueueError` if there was an error.
    ///
    /// # Errors
    ///
    /// This function may return a `QueueError::OperationError` if:
    /// - The command string has an invalid format
    /// - The command string contains an unknown gate type
    /// - The command string contains invalid parameters (e.g., non-numeric values for angles)
    /// - The command string contains invalid qubit indices
    #[allow(clippy::too_many_lines)]
    pub fn parse_command_to_builder(
        builder: &mut ByteMessageBuilder,
        cmd: &str,
    ) -> Result<(), QueueError> {
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        if parts.is_empty() {
            return Ok(());
        }

        match parts.first() {
            Some(&"RZ") => {
                if parts.len() >= 3 {
                    let theta = parts[1].parse::<f64>().map_err(|_| {
                        QueueError::OperationError(format!(
                            "Invalid angle in RZ command: {}",
                            parts[1]
                        ))
                    })?;
                    let qubit = parts[2].parse::<usize>().map_err(|_| {
                        QueueError::OperationError(format!(
                            "Invalid qubit in RZ command: {}",
                            parts[2]
                        ))
                    })?;
                    builder.add_rz(theta, &[qubit]);
                }
            }
            Some(&"R1XY") => {
                if parts.len() >= 4 {
                    let theta = parts[1].parse::<f64>().map_err(|_| {
                        QueueError::OperationError(format!(
                            "Invalid theta angle in R1XY command: {}",
                            parts[1]
                        ))
                    })?;
                    let phi = parts[2].parse::<f64>().map_err(|_| {
                        QueueError::OperationError(format!(
                            "Invalid phi angle in R1XY command: {}",
                            parts[2]
                        ))
                    })?;
                    let qubit = parts[3].parse::<usize>().map_err(|_| {
                        QueueError::OperationError(format!(
                            "Invalid qubit in R1XY command: {}",
                            parts[3]
                        ))
                    })?;
                    builder.add_r1xy(theta, phi, &[qubit]);
                }
            }
            Some(&"SZZ") => {
                if parts.len() >= 3 {
                    let qubit1 = parts[1].parse::<usize>().map_err(|_| {
                        QueueError::OperationError(format!(
                            "Invalid qubit1 in SZZ command: {}",
                            parts[1]
                        ))
                    })?;
                    let qubit2 = parts[2].parse::<usize>().map_err(|_| {
                        QueueError::OperationError(format!(
                            "Invalid qubit2 in SZZ command: {}",
                            parts[2]
                        ))
                    })?;
                    builder.add_szz(&[qubit1], &[qubit2]);
                }
            }
            Some(&"H") => {
                if parts.len() >= 2 {
                    let qubit = parts[1].parse::<usize>().map_err(|_| {
                        QueueError::OperationError(format!(
                            "Invalid qubit in H command: {}",
                            parts[1]
                        ))
                    })?;
                    builder.add_h(&[qubit]);
                }
            }
            Some(&"CX") => {
                if parts.len() >= 3 {
                    let control = parts[1].parse::<usize>().map_err(|_| {
                        QueueError::OperationError(format!(
                            "Invalid control qubit in CX command: {}",
                            parts[1]
                        ))
                    })?;
                    let target = parts[2].parse::<usize>().map_err(|_| {
                        QueueError::OperationError(format!(
                            "Invalid target qubit in CX command: {}",
                            parts[2]
                        ))
                    })?;
                    builder.add_cx(&[control], &[target]);
                }
            }
            Some(&"M") => {
                if parts.len() >= 3 {
                    let qubit = parts[1].parse::<usize>().map_err(|_| {
                        QueueError::OperationError(format!(
                            "Invalid qubit in M command: {}",
                            parts[1]
                        ))
                    })?;
                    let result_id = parts[2].parse::<usize>().map_err(|_| {
                        QueueError::OperationError(format!(
                            "Invalid result_id in M command: {}",
                            parts[2]
                        ))
                    })?;
                    builder.add_measurements(&[qubit], &[result_id]);
                }
            }
            Some(&"P") => {
                if parts.len() >= 2 {
                    let qubit = parts[1].parse::<usize>().map_err(|_| {
                        QueueError::OperationError(format!(
                            "Invalid qubit in P command: {}",
                            parts[1]
                        ))
                    })?;
                    builder.add_prep(&[qubit]);
                }
            }
            _ => {
                return Err(QueueError::OperationError(format!(
                    "Unknown command type: {}",
                    parts[0]
                )));
            }
        }

        Ok(())
    }

    /// Determine the message type by parsing the header
    ///
    /// This function parses the message header to determine the type of the message.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing the `MessageType` if successful, or a `QueueError` if there was an error.
    ///
    /// # Errors
    ///
    /// This function may return a `QueueError::OperationError` if:
    /// - The message is too small to contain a batch header
    /// - The batch header is invalid
    /// - The batch contains no messages
    /// - The message is too small to contain a message header
    /// - The message header contains an invalid message type
    pub fn message_type(&self) -> Result<MessageType, QueueError> {
        if self.bytes.len() < size_of::<BatchHeader>() {
            return Err(QueueError::OperationError(
                "Message too small for batch header".into(),
            ));
        }

        // Parse batch header
        let batch_header = *from_bytes::<BatchHeader>(&self.bytes[0..size_of::<BatchHeader>()]);
        if !batch_header.is_valid() {
            return Err(QueueError::OperationError("Invalid batch header".into()));
        }

        // Need at least one message to determine type
        if batch_header.msg_count == 0 {
            return Err(QueueError::OperationError(
                "Batch contains no messages".into(),
            ));
        }

        // Skip to first message header (after batch header)
        let msg_offset = size_of::<BatchHeader>();
        if self.bytes.len() < msg_offset + size_of::<MessageHeader>() {
            return Err(QueueError::OperationError(
                "Message too small for message header".into(),
            ));
        }

        // Parse message header
        let msg_header = *from_bytes::<MessageHeader>(
            &self.bytes[msg_offset..msg_offset + size_of::<MessageHeader>()],
        );
        msg_header
            .get_type()
            .map_err(|e| QueueError::OperationError(e.to_string()))
    }

    /// Check if this message is empty (contains no operations)
    ///
    /// This function checks if the message is empty, meaning it either contains a flush command
    /// or a batch with no operations.
    ///
    /// # Returns
    ///
    /// Returns a `Result` containing a boolean indicating whether the message is empty if successful,
    /// or a `QueueError` if there was an error.
    ///
    /// # Errors
    ///
    /// This function may return a `QueueError` if:
    /// - There is an error determining the message type
    /// - There is an error parsing the quantum operations in the message
    pub fn is_empty(&self) -> Result<bool, QueueError> {
        match self.message_type()? {
            MessageType::Flush => Ok(true),
            MessageType::BeginBatch => {
                // Check if this is a batch with no operations
                let commands = self.parse_quantum_operations()?;
                Ok(commands.is_empty())
            }
            _ => Ok(false),
        }
    }

    /// Parse quantum operations from this message
    pub fn parse_quantum_operations(&self) -> Result<Vec<QuantumGate>, QueueError> {
        if self.bytes.len() < size_of::<BatchHeader>() {
            return Err(QueueError::OperationError(
                "Message too small for batch header".into(),
            ));
        }

        // Parse batch header
        let batch_header = *from_bytes::<BatchHeader>(&self.bytes[0..size_of::<BatchHeader>()]);
        if !batch_header.is_valid() {
            return Err(QueueError::OperationError("Invalid batch header".into()));
        }

        let mut commands = Vec::new();
        let mut offset = size_of::<BatchHeader>();
        let mut in_command_batch = false;

        // Process each message
        for _ in 0..batch_header.msg_count {
            if offset + size_of::<MessageHeader>() > self.bytes.len() {
                break;
            }

            // Parse message header
            let msg_header = *from_bytes::<MessageHeader>(
                &self.bytes[offset..offset + size_of::<MessageHeader>()],
            );
            offset += size_of::<MessageHeader>();

            // Check if this is a quantum operations message
            if msg_header.msg_type == MessageType::BeginBatch as u8 {
                in_command_batch = true;
            } else if msg_header.msg_type == MessageType::EndBatch as u8 {
                // End of batch
                break;
            }

            // Skip to next message if not in a command batch
            if !in_command_batch {
                offset += msg_header.payload_size as usize;
                let padding = calc_padding(msg_header.payload_size as usize, 4);
                if padding > 0 {
                    offset += padding;
                }
                continue;
            }

            // Process payload based on message type
            match msg_header.msg_type {
                x if x == MessageType::QuantumGate as u8 => {
                    if offset + msg_header.payload_size as usize <= self.bytes.len() {
                        let payload =
                            &self.bytes[offset..offset + msg_header.payload_size as usize];
                        match Self::parse_quantum_gate(payload) {
                            Ok(cmd) => commands.push(cmd),
                            Err(e) => {
                                trace!("Error parsing quantum gate: {}", e);
                            }
                        }
                    }
                }
                x if x == MessageType::Measurement as u8 => {
                    if offset + msg_header.payload_size as usize <= self.bytes.len() {
                        let payload =
                            &self.bytes[offset..offset + msg_header.payload_size as usize];
                        match Self::parse_measurement(payload) {
                            Ok(cmd) => commands.push(cmd),
                            Err(e) => {
                                trace!("Error parsing measurement: {}", e);
                            }
                        }
                    }
                }
                _ => {}
            }

            // Move to next message
            offset += msg_header.payload_size as usize;
            let padding = calc_padding(msg_header.payload_size as usize, 4);
            if padding > 0 {
                offset += padding;
            }
        }

        Ok(commands)
    }

    /// Parse measurements from this message
    pub fn parse_measurements(&self) -> Result<Vec<(u32, u32)>, QueueError> {
        if self.bytes.len() < size_of::<BatchHeader>() {
            return Err(QueueError::OperationError(
                "Message too small for batch header".into(),
            ));
        }

        // Parse batch header
        let batch_header = *from_bytes::<BatchHeader>(&self.bytes[0..size_of::<BatchHeader>()]);
        if !batch_header.is_valid() {
            return Err(QueueError::OperationError("Invalid batch header".into()));
        }

        let mut measurements = Vec::new();
        let mut offset = size_of::<BatchHeader>();

        // Process each message
        for _ in 0..batch_header.msg_count {
            if offset + size_of::<MessageHeader>() > self.bytes.len() {
                break;
            }

            // Parse message header
            let msg_header = *from_bytes::<MessageHeader>(
                &self.bytes[offset..offset + size_of::<MessageHeader>()],
            );
            offset += size_of::<MessageHeader>();

            let msg_type = msg_header
                .get_type()
                .map_err(|e| QueueError::OperationError(e.to_string()))?;

            let payload_size = msg_header.payload_size as usize;
            let payload_end = offset + payload_size;

            if payload_end > self.bytes.len() {
                return Err(QueueError::OperationError(format!(
                    "Message payload extends beyond message bounds: offset={}, size={}, total_len={}",
                    offset,
                    payload_size,
                    self.bytes.len()
                )));
            }

            if msg_type == MessageType::MeasurementResult {
                // Process measurement result
                let payload = &self.bytes[offset..payload_end];
                if payload.len() >= size_of::<MeasurementResultHeader>() {
                    let result_header = *from_bytes::<MeasurementResultHeader>(
                        &payload[0..size_of::<MeasurementResultHeader>()],
                    );

                    // Return result_id and outcome as a tuple
                    measurements.push((result_header.result_id, result_header.outcome));
                }
            }

            // Move offset to next message, accounting for padding
            offset = payload_end;
            let padding = calc_padding(payload_size, 4);
            if padding > 0 {
                offset += padding;
            }
        }

        Ok(measurements)
    }

    /// Get measurement results as a vector of (`result_id`: usize, measurement: u32) pairs
    ///
    /// This is a convenience method that parses the measurement results from the message
    /// and returns them as a vector of tuples with the `result_id` converted to usize.
    ///
    /// # Returns
    ///
    /// A Result containing a vector of (`result_id`, measurement) pairs if successful,
    /// or a `QueueError` if there was an error parsing the message.
    pub fn measurement_results_as_vec(&self) -> Result<Vec<(usize, u32)>, QueueError> {
        let measurements = self.parse_measurements()?;

        // Convert result_ids from u32 to usize
        let converted = measurements
            .into_iter()
            .map(|(result_id, outcome)| (result_id as usize, outcome))
            .collect();

        Ok(converted)
    }

    /// Parse a quantum gate message payload
    fn parse_quantum_gate(payload: &[u8]) -> Result<QuantumGate, QueueError> {
        if payload.len() < size_of::<QuantumGateHeader>() {
            return Err(QueueError::OperationError(
                "Quantum gate message payload too small".into(),
            ));
        }

        let header = *from_bytes::<QuantumGateHeader>(&payload[0..size_of::<QuantumGateHeader>()]);
        let num_qubits = header.num_qubits as usize;
        let has_params = header.has_params != 0;

        // Calculate and validate sizes
        let qubits_size = num_qubits * size_of::<u32>();
        let minimum_size = size_of::<QuantumGateHeader>() + qubits_size;

        if payload.len() < minimum_size {
            return Err(QueueError::OperationError(
                "Quantum gate message payload too small for qubit indices".into(),
            ));
        }

        // Parse qubit indices
        let mut qubits = Vec::with_capacity(num_qubits);
        let qubits_offset = size_of::<QuantumGateHeader>();
        for i in 0..num_qubits {
            let qubit_offset = qubits_offset + i * size_of::<u32>();
            let qubit = u32::from_le_bytes([
                payload[qubit_offset],
                payload[qubit_offset + 1],
                payload[qubit_offset + 2],
                payload[qubit_offset + 3],
            ]) as usize;
            qubits.push(qubit);
        }

        // Parse parameters if present
        let mut params = Vec::new();
        let mut result_id = None;

        let gate_type = GateType::from(header.gate_type);

        if has_params {
            let params_offset = qubits_offset + qubits_size;
            match gate_type {
                GateType::RZ => {
                    if payload.len() >= params_offset + size_of::<f64>() {
                        let theta_bytes = &payload[params_offset..params_offset + size_of::<f64>()];
                        let theta = f64::from_le_bytes(theta_bytes[..8].try_into().unwrap());
                        params.push(theta);
                    } else {
                        return Err(QueueError::OperationError(
                            "Quantum gate message payload too small for RZ parameters".into(),
                        ));
                    }
                }
                GateType::R1XY => {
                    if payload.len() >= params_offset + 2 * size_of::<f64>() {
                        let theta_bytes = &payload[params_offset..params_offset + size_of::<f64>()];
                        let theta = f64::from_le_bytes(theta_bytes[..8].try_into().unwrap());
                        params.push(theta);

                        let phi_offset = params_offset + size_of::<f64>();
                        let phi_bytes = &payload[phi_offset..phi_offset + size_of::<f64>()];
                        let phi = f64::from_le_bytes(phi_bytes[..8].try_into().unwrap());
                        params.push(phi);
                    } else {
                        return Err(QueueError::OperationError(
                            "Quantum gate message payload too small for R1XY parameters".into(),
                        ));
                    }
                }
                GateType::RZZ => {
                    if payload.len() >= params_offset + size_of::<f64>() {
                        let theta_bytes = &payload[params_offset..params_offset + size_of::<f64>()];
                        let theta = f64::from_le_bytes(theta_bytes[..8].try_into().unwrap());
                        params.push(theta);
                    } else {
                        return Err(QueueError::OperationError(
                            "Quantum gate message payload too small for RZZ parameters".into(),
                        ));
                    }
                }
                GateType::Measure => {
                    if payload.len() >= params_offset + size_of::<u32>() {
                        let result_id_bytes =
                            &payload[params_offset..params_offset + size_of::<u32>()];
                        let result_id_value = u32::from_le_bytes([
                            result_id_bytes[0],
                            result_id_bytes[1],
                            result_id_bytes[2],
                            result_id_bytes[3],
                        ]) as usize;
                        result_id = Some(result_id_value);
                    } else {
                        return Err(QueueError::OperationError(
                            "Quantum gate message payload too small for Measure parameters".into(),
                        ));
                    }
                }
                _ => {}
            }
        }

        Ok(QuantumGate::new(gate_type, qubits, params, result_id))
    }

    /// Parse a measurement message payload
    fn parse_measurement(payload: &[u8]) -> Result<QuantumGate, QueueError> {
        if payload.len() < size_of::<MeasurementHeader>() {
            return Err(QueueError::OperationError(
                "Measurement message payload too small".into(),
            ));
        }

        let header = *from_bytes::<MeasurementHeader>(&payload[0..size_of::<MeasurementHeader>()]);
        let qubit = header.qubit as usize;
        let result_id = header.result_id as usize;

        Ok(QuantumGate::measure(qubit, result_id))
    }

    /// Creates an empty `ByteMessage`
    ///
    /// This method creates a minimal valid `ByteMessage` with no content.
    /// It's useful as a fallback when processing operations fails.
    ///
    /// # Returns
    /// A new empty `ByteMessage`
    #[must_use]
    pub fn create_empty() -> Self {
        Self { bytes: Vec::new() }
    }

    /// Create a record data message with key-value pair
    #[must_use]
    pub fn create_record_data(key: &str, value: f64) -> Self {
        let mut builder = ByteMessageBuilder::new();
        builder.add_record_data(key, value);
        builder.build()
    }

    /// Create a result record message
    #[must_use]
    pub fn create_result_record(result_id: usize, label: Option<&str>) -> Self {
        let mut builder = ByteMessageBuilder::new();
        builder.add_result_record(result_id, label);
        builder.build()
    }

    /// Create an info message
    #[must_use]
    pub fn create_info(message: &str) -> Self {
        let mut builder = ByteMessageBuilder::new();
        builder.add_info_message(message);
        builder.build()
    }

    /// Create a warning message
    #[must_use]
    pub fn create_warning(message: &str) -> Self {
        let mut builder = ByteMessageBuilder::new();
        builder.add_warning_message(message);
        builder.build()
    }

    /// Create an error message
    #[must_use]
    pub fn create_error(message: &str) -> Self {
        let mut builder = ByteMessageBuilder::new();
        builder.add_error_message(message);
        builder.build()
    }

    /// Create a debug message
    #[must_use]
    pub fn create_debug(message: &str) -> Self {
        let mut builder = ByteMessageBuilder::new();
        builder.add_debug_message(message);
        builder.build()
    }

    /// Parse measured qubits from this message
    ///
    /// This method extracts the qubit indices of measurements from the message.
    /// It returns a list of qubits that were measured, in the same order as
    /// the measurement results returned by `parse_measurements()`.
    ///
    /// # Returns
    ///
    /// A Result containing a vector of qubit indices if successful,
    /// or a `QueueError` if there was an error parsing the message.
    pub fn parse_measured_qubits(&self) -> Result<Vec<u32>, QueueError> {
        if self.bytes.is_empty() {
            return Ok(Vec::new());
        }

        let mut qubits = Vec::new();
        let mut offset = 0;

        while offset + size_of::<MessageHeader>() <= self.bytes.len() {
            // Read message header
            let msg_header = *from_bytes::<MessageHeader>(
                &self.bytes[offset..offset + size_of::<MessageHeader>()],
            );
            offset += size_of::<MessageHeader>();

            // Skip if not enough bytes for payload
            let payload_size = msg_header.payload_size as usize;
            let payload_end = offset + payload_size;

            if payload_end > self.bytes.len() {
                break;
            }

            // Convert the msg_type to MessageType
            if let Ok(msg_type) = msg_header.get_type() {
                if msg_type == MessageType::MeasurementResult {
                    // Process measurement result
                    let payload = &self.bytes[offset..payload_end];
                    if payload.len() >= size_of::<MeasurementResultHeader>() {
                        let result_header = *from_bytes::<MeasurementResultHeader>(
                            &payload[0..size_of::<MeasurementResultHeader>()],
                        );

                        // Since MeasurementResultHeader doesn't have a qubit field, we can't get it directly
                        // For now, we'll use the result_id as a placeholder - this needs to be fixed properly
                        // by tracking qubit-to-result mappings elsewhere
                        qubits.push(result_header.result_id);
                    }
                }
            }

            // Move offset to next message, accounting for padding
            offset = payload_end;
            let padding = calc_padding(payload_size, 4);
            if padding > 0 {
                offset += padding;
            }
        }

        Ok(qubits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::{Engine, quantum::StateVecEngine};

    #[test]
    fn test_bytemap_builder() {
        // Create a message with H and CX gates
        let mut builder = ByteMessage::quantum_operations_builder();
        builder.add_h(&[0]);
        builder.add_cx(&[0], &[1]);
        let message = builder.build();

        // Parse the message
        let parsed_commands = message.parse_quantum_operations().unwrap();
        assert_eq!(parsed_commands.len(), 2);
        assert_eq!(parsed_commands[0].gate_type, GateType::H);
        assert_eq!(parsed_commands[0].qubits, vec![0]);
        assert_eq!(parsed_commands[1].gate_type, GateType::CX);
        assert_eq!(parsed_commands[1].qubits, vec![0, 1]);
    }

    #[test]
    fn test_message_type() {
        // Create a flush message
        let flush_message = ByteMessage::create_flush();
        assert_eq!(flush_message.message_type().unwrap(), MessageType::Flush);

        // Create a quantum operations message
        let mut builder = ByteMessage::quantum_operations_builder();
        builder.add_h(&[0]);
        let quantum_message = builder.build();
        assert_eq!(
            quantum_message.message_type().unwrap(),
            MessageType::BeginBatch
        );

        // Create a measurement results message
        let mut builder = ByteMessage::measurement_results_builder();
        builder.add_measurement_results(&[0], &[1]);
        let results_message = builder.build();
        assert_eq!(
            results_message.message_type().unwrap(),
            MessageType::BeginBatch
        );
    }

    #[test]
    fn test_parse_measurements() {
        // Create a message with measurement results
        let mut builder = ByteMessage::measurement_results_builder();
        builder.add_measurement_results(&[0, 1], &[0, 1]);
        let message = builder.build();

        // Parse the measurements
        let measurements = message.parse_measurements().unwrap();
        assert_eq!(measurements.len(), 2);

        // The measurements are encoded as (result_id << 16) | outcome
        // So for result_id=0, outcome=0, we get 0
        // For result_id=1, outcome=1, we get 65537 (1 << 16 | 1)
        assert_eq!(measurements[0], (0, 0));
        assert_eq!(measurements[1], (1, 1));
    }

    #[test]
    fn test_measurement_results_as_vec() {
        // Create a message with measurement results
        let result_pairs = [(5, 0), (10, 1), (15, 0)];
        let message = ByteMessage::record_measurement_results(&result_pairs);

        // Get the results as a vector
        let results = message.measurement_results_as_vec().unwrap();

        // Verify the results match the input
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], (5, 0));
        assert_eq!(results[1], (10, 1));
        assert_eq!(results[2], (15, 0));

        // Verify the types are correct (usize, u32) by checking if they can be assigned to variables of those types
        let (result_id, outcome) = results[0];
        let _: usize = result_id; // This will fail to compile if result_id is not usize
        let _: u32 = outcome; // This will fail to compile if outcome is not u32
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
        builder.add_measurements(&[0], &[0]);

        // Measure qubit 1 with result_id 1
        builder.add_measurements(&[1], &[1]);

        let bell_circuit = builder.build();

        // Run the circuit multiple times and check the results
        let mut engine = StateVecEngine::new(2); // Create a simulator with 2 qubits

        for _ in 0..10 {
            // Reset the engine for each run
            engine.reset().unwrap();

            // Process the circuit
            let result_message = engine.process(bell_circuit.clone()).unwrap();

            // Get the measurement results as a vector
            let results = result_message.measurement_results_as_vec().unwrap();

            // Convert to booleans (0 -> false, 1 -> true)
            let q0_result = results
                .iter()
                .find(|(id, _)| *id == 0)
                .map(|(_, val)| *val != 0)
                .unwrap();
            let q1_result = results
                .iter()
                .find(|(id, _)| *id == 1)
                .map(|(_, val)| *val != 0)
                .unwrap();

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
        let empty_message = ByteMessage::builder().build();
        assert!(empty_message.is_empty().unwrap());

        // Create a non-empty message
        let non_empty_message = ByteMessage::quantum_operations_builder()
            .add_h(&[0])
            .build();
        assert!(!non_empty_message.is_empty().unwrap());
    }

    #[test]
    fn test_parse_command_to_builder() {
        // Test various commands including the new "P" command
        let commands = [
            "H 0", "CX 0 1", "RZ 0.5 2", "P 3", // Test the new P command
            "M 4 0",
        ];

        let message = ByteMessage::create_from_commands(&commands).unwrap();

        // Parse the quantum operations from the message
        let operations = message.parse_quantum_operations().unwrap();

        // We should have 5 operations
        assert_eq!(operations.len(), 5);

        // Check the P command was correctly parsed
        assert_eq!(operations[3].gate_type, GateType::Prep);
        assert_eq!(operations[3].qubits, vec![3]);
        assert!(operations[3].params.is_empty());
        assert_eq!(operations[3].result_id, None);
    }
}
