use crate::byte_message::builder::ByteMessageBuilder;
use crate::byte_message::protocol::{
    BatchHeader, GateHeader, MessageHeader, MessageType, OutcomeHeader, ReturnValueHeader,
    calc_padding,
};
use log::Level;
use log::trace;
use pecos_core::errors::PecosError;
use pecos_core::gate_type::GateType;
use pecos_core::gates::{Gate, GateAngles, GateParams, GateQubits};
use pecos_core::{Angle64, QubitId};
use std::fmt::Write as _;
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

    /// Check whether this message has zero bytes or a structurally valid zero-message batch.
    ///
    /// Payload contents and types do not determine emptiness.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid batch framing, without decoding payloads.
    pub fn is_empty(&self) -> Result<bool, PecosError> {
        if self.byte_len == 0 {
            return Ok(true);
        }
        let walker = MessageWalker::new(self)?;
        let empty = walker.msg_count() == 0;
        for message in walker {
            message?;
        }
        Ok(empty)
    }

    /// Parse every message as a quantum operation.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed batches, non-Gate messages, or invalid gates.
    pub fn quantum_ops(&self) -> Result<Vec<Gate>, PecosError> {
        let mut commands = Vec::new();
        self.quantum_ops_into(&mut commands)?;
        Ok(commands)
    }

    /// Parse every message as a quantum operation into an existing vector.
    ///
    /// This lets hot callers reuse vector capacity across repeated parses.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed batches, non-Gate messages, or invalid gates.
    pub fn quantum_ops_into(&self, commands: &mut Vec<Gate>) -> Result<(), PecosError> {
        let walker = MessageWalker::new(self)?;
        let trace_enabled = log::log_enabled!(Level::Trace);
        if trace_enabled {
            trace!("quantum_ops: Processing {} messages", walker.msg_count());
        }
        commands.clear();
        commands.reserve(walker.msg_count() as usize);
        for message in walker {
            let (index, msg_type, payload) = message?;
            Self::require_type(index, msg_type, MessageType::Gate)?;
            // Debug: dump payload bytes for RZ gates
            if trace_enabled && payload.len() >= size_of::<GateHeader>() {
                let header =
                    *bytemuck::from_bytes::<GateHeader>(&payload[0..size_of::<GateHeader>()]);
                if header.gate_type == GateType::RZ as u8 {
                    trace!("quantum_ops: RZ gate payload dump:");
                    trace!("  Total payload size: {} bytes", payload.len());
                    trace!(
                        "  Header: gate_type={}, num_qubits={}, has_params={}",
                        header.gate_type, header.num_qubits, header.has_params
                    );

                    // Dump raw bytes in hex only when trace logging is enabled.
                    let mut hex_bytes = String::with_capacity(payload.len().saturating_mul(3));
                    for (i, byte) in payload.iter().enumerate() {
                        if i > 0 {
                            hex_bytes.push(' ');
                        }
                        let _ = write!(&mut hex_bytes, "{byte:02x}");
                    }
                    trace!("  Raw bytes: {hex_bytes}");
                }
            }

            let gate = Self::parse_gate_command(payload)?;
            if trace_enabled {
                trace!("quantum_ops: Message {index} parsed as gate: {gate:?}");
            }
            commands.push(gate);
        }
        if trace_enabled {
            trace!("quantum_ops: Total gates parsed: {}", commands.len());
        }
        Ok(())
    }

    /// Parse every message as a measurement outcome.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed batches, non-Outcome messages, or invalid payload sizes.
    pub fn outcomes(&self) -> Result<Vec<u32>, PecosError> {
        let walker = MessageWalker::new(self)?;
        let mut measurements = Vec::new();
        for message in walker {
            let (index, msg_type, payload) = message?;
            Self::require_type(index, msg_type, MessageType::Outcome)?;
            let header = Self::read_payload_header::<OutcomeHeader>(index, payload, msg_type)?;
            measurements.push(header.outcome);
        }
        Ok(measurements)
    }

    /// Read a return value, or `None` for a zero-message batch.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed batches, non-ReturnValue messages, invalid payload
    /// sizes, or more than one return value.
    pub fn return_value(&self) -> Result<Option<i64>, PecosError> {
        let walker = MessageWalker::new(self)?;
        let mut value = None;
        for message in walker {
            let (index, msg_type, payload) = message?;
            Self::require_type(index, msg_type, MessageType::ReturnValue)?;
            let header = Self::read_payload_header::<ReturnValueHeader>(index, payload, msg_type)?;
            if value.is_some() {
                return Err(PecosError::Input(format!(
                    "Message {index}: expected at most one ReturnValue message"
                )));
            }
            value = Some(header.value);
        }
        Ok(value)
    }

    fn require_type(
        index: u32,
        found: MessageType,
        expected: MessageType,
    ) -> Result<(), PecosError> {
        if found != expected {
            return Err(PecosError::Input(format!(
                "Message {index}: expected {expected:?}, found {found:?}"
            )));
        }
        Ok(())
    }

    fn read_payload_header<T: bytemuck::Pod>(
        index: u32,
        payload: &[u8],
        msg_type: MessageType,
    ) -> Result<T, PecosError> {
        let expected = size_of::<T>();
        if payload.len() != expected {
            return Err(PecosError::Input(format!(
                "Message {index}: {msg_type:?} payload size must be {expected}, found {}",
                payload.len()
            )));
        }
        Ok(bytemuck::pod_read_unaligned::<T>(payload))
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
    fn parse_qubit_indices(payload: &[u8], qubits_offset: usize, num_qubits: usize) -> GateQubits {
        let mut qubits = GateQubits::with_capacity(num_qubits);
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
        param_count: usize,
    ) -> Result<(GateAngles, GateParams), PecosError> {
        let trace_enabled = log::log_enabled!(Level::Trace);
        if param_count == 0 {
            return Ok((GateAngles::new(), GateParams::new()));
        }

        if trace_enabled {
            trace!("parse_gate_parameters: Gate {gate_type:?} requires {param_count} parameters");
        }

        let angle_count = if gate_type == GateType::Custom {
            param_count
        } else {
            gate_type.angle_arity()
        };
        let mut angles = GateAngles::with_capacity(angle_count);
        let mut params = GateParams::with_capacity(param_count.saturating_sub(angle_count));
        for i in 0..param_count {
            let param_offset = params_offset + i * size_of::<f64>();
            let param = Self::parse_f64_param(payload, param_offset);
            if trace_enabled {
                trace!("parse_gate_parameters: Parameter {i} at offset {param_offset}: {param}");
            }
            if i < angle_count {
                if !param.is_finite() {
                    return Err(PecosError::Input("gate angle must be finite".into()));
                }
                angles.push(Angle64::from_radians(param));
            } else {
                params.push(param);
            }
        }

        // Special logging for RZ gate parameters
        if trace_enabled && matches!(gate_type, GateType::RZ) && !angles.is_empty() {
            trace!(
                "parse_gate_parameters: RZ angle parsed as {} radians ({} degrees)",
                angles[0].to_radians(),
                angles[0].to_radians().to_degrees()
            );
        }

        Ok((angles, params))
    }

    /// Validate that the payload has exactly the parameter bytes required by the gate.
    fn validate_params_size(
        payload: &[u8],
        params_offset: usize,
        required_size: usize,
        gate_type: GateType,
    ) -> Result<usize, PecosError> {
        let received_size = payload.len().saturating_sub(params_offset);
        if gate_type == GateType::Custom {
            if !received_size.is_multiple_of(size_of::<f64>()) {
                return Err(PecosError::Input(format!(
                    "Gate Custom received {received_size} parameter bytes, which is not a whole number of f64 parameters"
                )));
            }
            return Ok(received_size / size_of::<f64>());
        }

        if received_size != required_size {
            let expected_params = gate_type.classical_arity();
            return Err(PecosError::Input(format!(
                "Gate {gate_type:?} expected {expected_params} parameters ({required_size} bytes), received {received_size} parameter bytes"
            )));
        }
        Ok(gate_type.classical_arity())
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
        let trace_enabled = log::log_enabled!(Level::Trace);
        Self::validate_gate_payload_size(payload)?;

        // Parse gate header - guaranteed aligned since payload starts at aligned boundary
        let header = *bytemuck::from_bytes::<GateHeader>(&payload[0..size_of::<GateHeader>()]);
        let num_qubits = header.num_qubits as usize;
        let has_params = header.has_params != 0;
        let gate_type = GateType::try_from(header.gate_type).map_err(PecosError::Input)?;
        if gate_type == GateType::Channel {
            return Err(PecosError::Input(
                "Channel gates carry typed payloads and cannot be encoded in ByteMessage gate commands"
                    .to_string(),
            ));
        }

        if trace_enabled {
            trace!(
                "parse_gate_command: Parsing gate type {gate_type:?}, num_qubits: {num_qubits}, has_params: {has_params}"
            );
        }

        // Calculate sizes
        let qubits_byte_size = num_qubits * size_of::<u32>();
        let qubits_offset = size_of::<GateHeader>();

        Self::validate_qubit_indices_size(payload, qubits_offset, qubits_byte_size)?;

        // Parse qubit indices directly to QubitId
        let qubits = Self::parse_qubit_indices(payload, qubits_offset, num_qubits);

        let params_offset = qubits_offset + qubits_byte_size;
        let required_params_size = gate_type.classical_arity() * size_of::<f64>();
        let param_count =
            Self::validate_params_size(payload, params_offset, required_params_size, gate_type)?;

        if trace_enabled {
            trace!("parse_gate_command: Parsed qubits: {qubits:?}");
        }

        // Parse parameters if present
        // The wire format stores all classical parameters as f64, with angles first (in radians)
        let (angles, params) = if has_params || (gate_type == GateType::Custom && param_count > 0) {
            let parsed =
                Self::parse_gate_parameters(payload, params_offset, gate_type, param_count)?;
            if trace_enabled {
                trace!(
                    "parse_gate_command: Parsed parameters: angles={:?}, params={:?}",
                    parsed.0, parsed.1
                );
            }
            parsed
        } else {
            (GateAngles::new(), GateParams::new())
        };

        // Special logging for RZ gates
        if trace_enabled && matches!(gate_type, GateType::RZ) {
            trace!(
                "parse_gate_command: RZ gate parsed with angle: {:?}, qubit: {:?}",
                angles.first(),
                qubits.first()
            );
        }

        let gate = Gate::try_new(gate_type, angles, params, qubits)
            .map_err(|err| PecosError::Input(format!("Invalid gate command payload: {err}")))?;
        gate.validate().map_err(|err| {
            PecosError::Input(format!(
                "Invalid gate command payload for {gate_type:?}: {err}"
            ))
        })?;
        Ok(gate)
    }

    // The parse_simple_measurement method has been removed as part of simplifying the protocol.
    // All measurements are now handled as regular gates through parse_gate_command.
}

/// Walk batch framing without interpreting message payloads.
struct MessageWalker<'a> {
    bytes: &'a [u8],
    msg_count: u32,
    index: u32,
    offset: usize,
}

impl<'a> MessageWalker<'a> {
    fn new(message: &'a ByteMessage) -> Result<Self, PecosError> {
        let header = message.parse_batch_header()?;
        let bytes = message.as_bytes();
        if header.total_size as usize != bytes.len() {
            return Err(PecosError::Input(format!(
                "Batch: total_size {} does not match byte length {}",
                header.total_size,
                bytes.len()
            )));
        }
        let max_count = (bytes.len() - size_of::<BatchHeader>()) / size_of::<MessageHeader>();
        if header.msg_count as usize > max_count {
            return Err(PecosError::Input(format!(
                "Batch: msg_count {} exceeds capacity {max_count}",
                header.msg_count
            )));
        }
        if header.msg_count == 0 && bytes.len() != size_of::<BatchHeader>() {
            return Err(PecosError::Input(
                "Batch: trailing bytes after zero-count batch".into(),
            ));
        }
        Ok(Self {
            bytes,
            msg_count: header.msg_count,
            index: 0,
            offset: size_of::<BatchHeader>(),
        })
    }

    fn msg_count(&self) -> u32 {
        self.msg_count
    }

    fn read_message(&mut self) -> Result<(u32, MessageType, &'a [u8]), PecosError> {
        let index = self.index;
        let error = |condition| PecosError::Input(format!("Message {index}: {condition}"));
        let payload_start = self
            .offset
            .checked_add(size_of::<MessageHeader>())
            .ok_or_else(|| error("message header offset overflow"))?;
        if payload_start > self.bytes.len() {
            return Err(error("message header extends beyond buffer"));
        }
        let header =
            bytemuck::pod_read_unaligned::<MessageHeader>(&self.bytes[self.offset..payload_start]);
        let msg_type = header.get_type().map_err(error)?;
        let payload_end = payload_start
            .checked_add(header.payload_size as usize)
            .ok_or_else(|| error("payload offset overflow"))?;
        if payload_end > self.bytes.len() {
            return Err(error("payload extends beyond buffer"));
        }
        let next_offset = payload_end
            .checked_add(calc_padding(payload_end, 4))
            .ok_or_else(|| error("padded offset overflow"))?;
        if index == self.msg_count - 1 && self.bytes.len() > next_offset {
            return Err(error("trailing bytes after final payload padding"));
        }
        self.offset = next_offset;
        Ok((index, msg_type, &self.bytes[payload_start..payload_end]))
    }
}

impl<'a> Iterator for MessageWalker<'a> {
    type Item = Result<(u32, MessageType, &'a [u8]), PecosError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.msg_count {
            return None;
        }
        let result = self.read_message();
        if result.is_err() {
            self.index = self.msg_count;
        } else {
            self.index += 1;
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Engine;
    use crate::byte_message::protocol::MessageFlags;
    use crate::quantum::StateVecEngine;
    use pecos_core::QubitId;

    fn patch_word(bytes: &mut [u8], offset: usize, value: u32) {
        bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
    }

    fn patched_message(mut bytes: Vec<u8>) -> ByteMessage {
        let len = u32::try_from(bytes.len()).unwrap();
        patch_word(&mut bytes, 12, len);
        ByteMessage::new(&bytes)
    }

    fn assert_input<T: std::fmt::Debug>(result: Result<T, PecosError>, expected: &str) {
        match result {
            Err(PecosError::Input(message)) => assert_eq!(message, expected),
            other => panic!("expected Input({expected:?}), got {other:?}"),
        }
    }

    #[test]
    fn strict_unaligned_payload_headers() {
        let mut storage = [0_u64; 2];
        let bytes = bytemuck::cast_slice_mut::<u64, u8>(&mut storage);
        bytes[4..12].copy_from_slice(&(-42_i64).to_le_bytes());
        let value = ByteMessage::read_payload_header::<ReturnValueHeader>(
            0,
            &bytes[4..12],
            MessageType::ReturnValue,
        )
        .unwrap();
        assert_eq!(value.value, -42);
        bytes[1..5].copy_from_slice(&1_u32.to_le_bytes());
        let outcome = ByteMessage::read_payload_header::<OutcomeHeader>(
            0,
            &bytes[1..5],
            MessageType::Outcome,
        )
        .unwrap();
        assert_eq!(outcome.outcome, 1);
    }

    #[test]
    fn strict_batch_header() {
        let mut short = ByteMessage::create_empty().into_bytes();
        patch_word(&mut short, 12, 15);
        short.truncate(15);
        assert_input(
            ByteMessage::new(&short).quantum_ops(),
            "Message too small for batch header",
        );
        let mut bytes = ByteMessage::create_empty().into_bytes();
        bytes[0] = 0;
        assert_input(
            ByteMessage::new(&bytes).quantum_ops(),
            "Invalid batch header",
        );
        bytes = ByteMessage::create_empty().into_bytes();
        bytes[4] = 255;
        assert_input(
            ByteMessage::new(&bytes).quantum_ops(),
            "Invalid batch header",
        );
    }

    #[test]
    fn strict_total_size() {
        let mut bytes = ByteMessage::create_empty().into_bytes();
        patch_word(&mut bytes, 12, 17);
        assert_input(
            ByteMessage::new(&bytes).quantum_ops(),
            "Batch: total_size 17 does not match byte length 16",
        );
    }

    #[test]
    fn strict_count_before_reserve() {
        let mut bytes = ByteMessage::create_empty().into_bytes();
        patch_word(&mut bytes, 8, u32::MAX);
        assert_input(
            ByteMessage::new(&bytes).quantum_ops(),
            "Batch: msg_count 4294967295 exceeds capacity 0",
        );
    }

    #[test]
    fn strict_later_header() {
        let mut bytes = ByteMessage::builder().h(&[0]).x(&[1]).build().into_bytes();
        bytes.truncate(36);
        let message = patched_message(bytes);
        assert_input(
            message.quantum_ops(),
            "Message 1: message header extends beyond buffer",
        );
    }

    #[test]
    fn strict_unknown_type() {
        let mut bytes = ByteMessage::builder().h(&[0]).build().into_bytes();
        bytes[16] = 255;
        assert_input(
            ByteMessage::new(&bytes).quantum_ops(),
            "Message 0: Unknown message type",
        );
    }

    #[test]
    fn strict_payload_bounds() {
        let mut bytes = ByteMessage::builder().h(&[0]).build().into_bytes();
        patch_word(&mut bytes, 20, 9);
        assert_input(
            ByteMessage::new(&bytes).quantum_ops(),
            "Message 0: payload extends beyond buffer",
        );
    }

    #[test]
    fn strict_final_trailing_bytes() {
        let mut bytes = ByteMessage::builder()
            .add_return_value(42)
            .build()
            .into_bytes();
        bytes.push(0);
        let message = patched_message(bytes);
        let mut walker = MessageWalker::new(&message).unwrap();
        assert_input(
            walker.next().unwrap(),
            "Message 0: trailing bytes after final payload padding",
        );
        assert_input(
            message.return_value(),
            "Message 0: trailing bytes after final payload padding",
        );
    }

    #[test]
    fn strict_zero_count_trailing_bytes() {
        let mut bytes = ByteMessage::create_empty().into_bytes();
        bytes.push(0);
        let message = patched_message(bytes);
        assert_input(
            MessageWalker::new(&message).map(|walker| walker.msg_count()),
            "Batch: trailing bytes after zero-count batch",
        );
        assert_input(
            message.is_empty(),
            "Batch: trailing bytes after zero-count batch",
        );
    }

    #[test]
    fn strict_fixed_payload_sizes() {
        for (kind, expected) in [(MessageType::Outcome, 4), (MessageType::ReturnValue, 8)] {
            for size in [expected - 1, expected + 1] {
                let mut builder = ByteMessage::builder();
                match kind {
                    MessageType::Outcome => {
                        builder.add_outcomes(&[1]);
                    }
                    MessageType::ReturnValue => {
                        builder.add_return_value(42);
                    }
                    MessageType::Gate => unreachable!(),
                }
                let mut bytes = builder.build().into_bytes();
                bytes.resize(24 + size, 0);
                patch_word(&mut bytes, 20, u32::try_from(size).unwrap());
                let message = patched_message(bytes);
                let error =
                    format!("Message 0: {kind:?} payload size must be {expected}, found {size}");
                match kind {
                    MessageType::Outcome => assert_input(message.outcomes(), &error),
                    MessageType::ReturnValue => assert_input(message.return_value(), &error),
                    MessageType::Gate => unreachable!(),
                }
                assert!(!message.is_empty().unwrap());
            }
        }
    }

    #[test]
    fn strict_wrong_types() {
        let gate = ByteMessage::builder().h(&[0]).build();
        let outcome = ByteMessage::builder().add_outcomes(&[1]).build();
        let value = ByteMessage::builder().add_return_value(42).build();
        assert_input(
            outcome.quantum_ops(),
            "Message 0: expected Gate, found Outcome",
        );
        assert_input(
            value.quantum_ops(),
            "Message 0: expected Gate, found ReturnValue",
        );
        assert_input(gate.outcomes(), "Message 0: expected Outcome, found Gate");
        assert_input(
            value.outcomes(),
            "Message 0: expected Outcome, found ReturnValue",
        );
        assert_input(
            gate.return_value(),
            "Message 0: expected ReturnValue, found Gate",
        );
        assert_input(
            outcome.return_value(),
            "Message 0: expected ReturnValue, found Outcome",
        );
    }

    fn mixed_batch(first: &ByteMessage, second: &ByteMessage) -> ByteMessage {
        let mut bytes = first.as_bytes().to_vec();
        bytes.extend_from_slice(&second.as_bytes()[16..]);
        patch_word(&mut bytes, 8, 2);
        patched_message(bytes)
    }

    #[test]
    fn strict_mixed_types() {
        let gate = ByteMessage::builder().h(&[0]).build();
        let outcome = ByteMessage::builder().add_outcomes(&[1]).build();
        let value = ByteMessage::builder().add_return_value(42).build();
        assert_input(
            mixed_batch(&gate, &outcome).quantum_ops(),
            "Message 1: expected Gate, found Outcome",
        );
        assert_input(
            mixed_batch(&outcome, &gate).outcomes(),
            "Message 1: expected Outcome, found Gate",
        );
        assert_input(
            mixed_batch(&value, &gate).return_value(),
            "Message 1: expected ReturnValue, found Gate",
        );
        assert!(!mixed_batch(&gate, &outcome).is_empty().unwrap());
    }

    #[test]
    fn strict_return_value_count_and_whole_batch() {
        assert_eq!(ByteMessage::create_empty().return_value().unwrap(), None);
        let value = ByteMessage::builder().add_return_value(-42).build();
        assert_eq!(value.return_value().unwrap(), Some(-42));
        let two = mixed_batch(&value, &value);
        assert_input(
            two.return_value(),
            "Message 1: expected at most one ReturnValue message",
        );
        let mut bytes = two.into_bytes();
        bytes[32] = 255;
        assert_input(
            ByteMessage::new(&bytes).return_value(),
            "Message 1: Unknown message type",
        );
    }

    #[test]
    fn strict_is_empty_checks_only_framing() {
        assert!(ByteMessage::new(&[]).is_empty().unwrap());
        let empty = ByteMessage::create_empty();
        assert!(empty.is_empty().unwrap());
        assert_eq!(empty.quantum_ops().unwrap(), [] as [Gate; 0]);
        assert_eq!(empty.outcomes().unwrap(), [] as [u32; 0]);
        let gate = ByteMessage::builder().h(&[0]).build();
        assert!(!gate.is_empty().unwrap());
        assert!(
            !ByteMessage::builder()
                .add_outcomes(&[1])
                .build()
                .is_empty()
                .unwrap()
        );
        let mut bytes = gate.into_bytes();
        bytes[24] = 255; // Gate decoding is deliberately outside is_empty's contract.
        assert!(!ByteMessage::new(&bytes).is_empty().unwrap());
        bytes[16] = 255;
        assert_input(
            ByteMessage::new(&bytes).is_empty(),
            "Message 0: Unknown message type",
        );
    }

    #[test]
    fn strict_padding_boundaries() {
        let outcome = ByteMessage::builder().add_outcomes(&[1, 0]).build();
        let mut bytes = outcome.into_bytes();
        patch_word(&mut bytes, 20, 3); // Next header remains at the rounded-up offset, 28.
        let message = ByteMessage::new(&bytes);
        let records = MessageWalker::new(&message)
            .unwrap()
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].2.len(), 3);
        assert_eq!(records[1].2, [0, 0, 0, 0]);
        bytes.truncate(27);
        patch_word(&mut bytes, 8, 1);
        for padding in 0..=1 {
            let mut padded = bytes.clone();
            padded.resize(27 + padding, 0);
            assert!(!patched_message(padded).is_empty().unwrap());
        }
        bytes.resize(29, 0);
        assert_input(
            patched_message(bytes).is_empty(),
            "Message 0: trailing bytes after final payload padding",
        );
    }

    #[test]
    fn strict_quantum_ops_reuses_capacity() {
        let message = ByteMessage::builder().h(&[0]).build();
        let mut commands = Vec::with_capacity(8);
        commands.push(Gate::x(&[1]));
        let capacity = commands.capacity();
        message.quantum_ops_into(&mut commands).unwrap();
        assert_eq!(commands, [Gate::h(&[0])]);
        assert_eq!(commands.capacity(), capacity);
        ByteMessage::create_empty()
            .quantum_ops_into(&mut commands)
            .unwrap();
        assert_eq!(commands, [] as [Gate; 0]);
        assert_eq!(commands.capacity(), capacity);
    }

    #[test]
    fn quantum_ops_rejects_surplus_gate_parameter_bytes() {
        let header = GateHeader {
            gate_type: GateType::RZ as u8,
            num_qubits: 1,
            has_params: 1,
            reserved: 0,
        };
        let mut payload = Vec::new();
        payload.extend_from_slice(bytemuck::bytes_of(&header));
        payload.extend_from_slice(&0_u32.to_le_bytes());
        payload.extend_from_slice(&0.5_f64.to_le_bytes());
        payload.extend_from_slice(&0.25_f64.to_le_bytes());

        let mut builder = ByteMessageBuilder::new();
        builder.add_message(MessageType::Gate, &payload, MessageFlags::NONE);
        let err = builder
            .build()
            .quantum_ops()
            .expect_err("surplus angle bytes must be rejected");

        assert!(err.to_string().contains("Gate RZ expected 1 parameters"));
        assert!(err.to_string().contains("received 16 parameter bytes"));
    }

    #[test]
    fn quantum_ops_rejects_retired_and_unknown_gate_ids_without_panicking() {
        for gate_type in [70_u8, 254_u8] {
            let header = GateHeader {
                gate_type,
                num_qubits: 2,
                has_params: 0,
                reserved: 0,
            };
            let mut payload = Vec::new();
            payload.extend_from_slice(bytemuck::bytes_of(&header));
            payload.extend_from_slice(&0_u32.to_le_bytes());
            payload.extend_from_slice(&1_u32.to_le_bytes());
            let mut builder = ByteMessageBuilder::new();
            builder.add_message(MessageType::Gate, &payload, MessageFlags::NONE);

            let error = builder
                .build()
                .quantum_ops()
                .expect_err("unknown gate id must return a structured error");
            assert!(
                matches!(error, PecosError::Input(_)),
                "unexpected error: {error}"
            );
            assert!(error.to_string().contains(&gate_type.to_string()));
        }
    }

    #[test]
    fn quantum_ops_round_trips_custom_gate_angles() {
        let original = Gate::new(
            GateType::Custom,
            vec![Angle64::from_radians(0.5), Angle64::from_radians(0.25)],
            Vec::<f64>::new(),
            vec![QubitId(0), QubitId(1)],
        );
        let mut builder = ByteMessageBuilder::new();
        builder.add_gate_command(&original);

        let decoded = builder
            .build()
            .quantum_ops()
            .expect("custom gate angles should decode");

        assert_eq!(decoded, [original]);
    }

    #[test]
    fn test_bytemap_builder() {
        // Create a message with H and CX gates
        let mut builder = ByteMessage::quantum_operations_builder();
        builder.h(&[0]);
        builder.cx(&[(0, 1)]);
        let message = builder.build();

        // Parse the message
        let parsed_commands = message.quantum_ops().unwrap();
        assert_eq!(parsed_commands.len(), 2);
        assert_eq!(parsed_commands[0].gate_type, GateType::H);
        assert_eq!(parsed_commands[0].qubits.as_slice(), &[QubitId(0)]);
        assert_eq!(parsed_commands[1].gate_type, GateType::CX);
        assert_eq!(
            parsed_commands[1].qubits.as_slice(),
            &[QubitId(0), QubitId(1)]
        );
    }

    #[test]
    fn test_raw_channel_gate_message_is_rejected() {
        use crate::byte_message::protocol::{GateHeader, MessageFlags, MessageType};

        let header = GateHeader {
            gate_type: GateType::Channel as u8,
            num_qubits: 1,
            has_params: 0,
            reserved: 0,
        };
        let mut payload = Vec::new();
        payload.extend_from_slice(bytemuck::bytes_of(&header));
        payload.extend_from_slice(&0u32.to_le_bytes());

        let mut builder = ByteMessage::quantum_operations_builder();
        builder.add_message(MessageType::Gate, &payload, MessageFlags::NONE);
        let message = builder.build();

        let err = message
            .quantum_ops()
            .expect_err("ByteMessage cannot carry typed channel payloads");

        assert!(
            err.to_string()
                .contains("Channel gates carry typed payloads")
        );
    }

    #[test]
    fn test_raw_invalid_gate_payload_is_rejected_after_parse() {
        use crate::byte_message::protocol::{GateHeader, MessageFlags, MessageType};

        let header = GateHeader {
            gate_type: GateType::CX as u8,
            num_qubits: 2,
            has_params: 0,
            reserved: 0,
        };
        let mut payload = Vec::new();
        payload.extend_from_slice(bytemuck::bytes_of(&header));
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&0u32.to_le_bytes());

        let mut builder = ByteMessage::quantum_operations_builder();
        builder.add_message(MessageType::Gate, &payload, MessageFlags::NONE);
        let message = builder.build();

        let err = message
            .quantum_ops()
            .expect_err("raw CX payload cannot use the same qubit twice");

        assert!(
            err.to_string().contains("Invalid gate command payload"),
            "unexpected error: {err}"
        );
        assert!(
            err.to_string().contains("requires distinct qubits"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn test_message_type() {
        // Create an empty message
        let empty_message = ByteMessage::create_empty();

        // Empty message should be parseable
        assert!(empty_message.is_empty().unwrap());

        // Create a quantum operations message
        let mut builder = ByteMessage::quantum_operations_builder();
        builder.h(&[0]);
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
        builder.h(&[0]);

        // Apply CX with control=0, target=1
        builder.cx(&[(0, 1)]);

        // Measure qubit 0 with result_id 0
        builder.mz(&[0]);

        // Measure qubit 1 with result_id 1
        builder.mz(&[1]);

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
        let non_empty_message = ByteMessage::quantum_operations_builder().h(&[0]).build();
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
        builder.mz(&[3]); // First: measure qubit 3
        builder.mz(&[1]); // Second: measure qubit 1
        builder.mz(&[4]); // Third: measure qubit 4
        builder.mz(&[0]); // Fourth: measure qubit 0
        builder.mz(&[2]); // Fifth: measure qubit 2

        let message = builder.build();

        // Parse operations back
        let operations = message.quantum_ops().unwrap();

        // Verify we have 5 measurement operations in the correct order
        assert_eq!(operations.len(), 5);

        assert_eq!(operations[0].gate_type, GateType::MZ);
        assert_eq!(operations[0].qubits.as_slice(), &[QubitId(3)]);

        assert_eq!(operations[1].gate_type, GateType::MZ);
        assert_eq!(operations[1].qubits.as_slice(), &[QubitId(1)]);

        assert_eq!(operations[2].gate_type, GateType::MZ);
        assert_eq!(operations[2].qubits.as_slice(), &[QubitId(4)]);

        assert_eq!(operations[3].gate_type, GateType::MZ);
        assert_eq!(operations[3].qubits.as_slice(), &[QubitId(0)]);

        assert_eq!(operations[4].gate_type, GateType::MZ);
        assert_eq!(operations[4].qubits.as_slice(), &[QubitId(2)]);
    }
}
