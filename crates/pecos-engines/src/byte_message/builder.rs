//! Message builder for creating byte-encoded messages
//!
//! This module provides utilities for constructing binary messages
//! according to the byte protocol.

use crate::byte_message::GateType;
use crate::byte_message::message::ByteMessage;
use crate::byte_message::protocol::{
    BatchHeader, GateHeader, MessageFlags, MessageHeader, MessageType, OutcomeHeader,
    ReturnValueHeader, calc_padding,
};
use bytemuck::bytes_of;
use pecos_core::Angle64;
use pecos_core::gates::Gate;
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

    fn prepare_message(&mut self, msg_type: MessageType, payload_size: usize, flags: MessageFlags) {
        match msg_type {
            MessageType::Gate => {
                assert!(
                    !(self.mode == BuilderMode::MeasurementOutcomes),
                    "Cannot mix quantum operations and measurement outcomes in the same message"
                );
                if self.mode == BuilderMode::Empty {
                    self.mode = BuilderMode::QuantumOperations;
                }
            }
            MessageType::Outcome => {
                assert!(
                    !(self.mode == BuilderMode::QuantumOperations
                        || self.mode == BuilderMode::ReturnValue),
                    "Cannot mix measurement outcomes with other message types"
                );
                self.mode = BuilderMode::MeasurementOutcomes;
            }
            MessageType::ReturnValue => {
                assert!(
                    self.mode == BuilderMode::Empty || self.mode == BuilderMode::ReturnValue,
                    "Cannot mix return values with other message types"
                );
                self.mode = BuilderMode::ReturnValue;
            }
        }

        self.add_padding(4);

        let payload_size = u32::try_from(payload_size).unwrap_or_else(|_| {
            log::warn!("Payload size exceeds u32::MAX, using maximum value");
            u32::MAX
        });
        let header = MessageHeader::new(msg_type, payload_size, flags);
        self.buffer.extend_from_slice(bytes_of(&header));
        self.msg_count += 1;
    }

    fn add_gate_parts_from_usizes<I>(
        &mut self,
        gate_type: GateType,
        num_qubits: usize,
        qubits: I,
        angles: &[Angle64],
        params: &[f64],
    ) -> &mut Self
    where
        I: IntoIterator<Item = usize>,
    {
        assert!(
            gate_type != GateType::Channel,
            "Channel gates carry typed payloads and cannot be encoded in ByteMessage gate commands"
        );
        let payload_size = size_of::<GateHeader>()
            + num_qubits * size_of::<u32>()
            + (angles.len() + params.len()) * size_of::<f64>();

        self.prepare_message(MessageType::Gate, payload_size, MessageFlags::NONE);

        let header = GateHeader {
            gate_type: gate_type as u8,
            num_qubits: u8::try_from(num_qubits).expect("Too many qubits for gate"),
            has_params: u8::from(!angles.is_empty() || !params.is_empty()),
            reserved: 0,
        };
        self.buffer.extend_from_slice(bytes_of(&header));

        for qubit in qubits {
            let qubit_u32 = u32::try_from(qubit).expect("Qubit index too large");
            self.buffer.extend_from_slice(&qubit_u32.to_le_bytes());
        }

        for angle in angles {
            self.buffer
                .extend_from_slice(&angle.to_radians().to_le_bytes());
        }

        for param in params {
            self.buffer.extend_from_slice(&param.to_le_bytes());
        }

        self
    }

    fn add_gate_parts(
        &mut self,
        gate_type: GateType,
        qubits: &[usize],
        angles: &[Angle64],
        params: &[f64],
    ) -> &mut Self {
        self.add_gate_parts_from_usizes(
            gate_type,
            qubits.len(),
            qubits.iter().copied(),
            angles,
            params,
        )
    }

    #[inline]
    fn add_single_qubit_gate_parts(
        &mut self,
        gate_type: GateType,
        qubit: usize,
        angles: &[Angle64],
        params: &[f64],
    ) -> &mut Self {
        let payload_size = size_of::<GateHeader>()
            + size_of::<u32>()
            + (angles.len() + params.len()) * size_of::<f64>();

        self.prepare_message(MessageType::Gate, payload_size, MessageFlags::NONE);

        let header = GateHeader {
            gate_type: gate_type as u8,
            num_qubits: 1,
            has_params: u8::from(!angles.is_empty() || !params.is_empty()),
            reserved: 0,
        };
        self.buffer.extend_from_slice(bytes_of(&header));

        let qubit_u32 = u32::try_from(qubit).expect("Qubit index too large");
        self.buffer.extend_from_slice(&qubit_u32.to_le_bytes());

        for angle in angles {
            self.buffer
                .extend_from_slice(&angle.to_radians().to_le_bytes());
        }

        for param in params {
            self.buffer.extend_from_slice(&param.to_le_bytes());
        }

        self
    }

    #[inline]
    fn add_two_qubit_gate_parts(
        &mut self,
        gate_type: GateType,
        qubit0: usize,
        qubit1: usize,
        angles: &[Angle64],
        params: &[f64],
    ) -> &mut Self {
        let payload_size = size_of::<GateHeader>()
            + 2 * size_of::<u32>()
            + (angles.len() + params.len()) * size_of::<f64>();

        self.prepare_message(MessageType::Gate, payload_size, MessageFlags::NONE);

        let header = GateHeader {
            gate_type: gate_type as u8,
            num_qubits: 2,
            has_params: u8::from(!angles.is_empty() || !params.is_empty()),
            reserved: 0,
        };
        self.buffer.extend_from_slice(bytes_of(&header));

        let qubit0_u32 = u32::try_from(qubit0).expect("Qubit index too large");
        let qubit1_u32 = u32::try_from(qubit1).expect("Qubit index too large");
        self.buffer.extend_from_slice(&qubit0_u32.to_le_bytes());
        self.buffer.extend_from_slice(&qubit1_u32.to_le_bytes());

        for angle in angles {
            self.buffer
                .extend_from_slice(&angle.to_radians().to_le_bytes());
        }

        for param in params {
            self.buffer.extend_from_slice(&param.to_le_bytes());
        }

        self
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
        self.prepare_message(msg_type, payload.len(), flags);
        self.buffer.extend_from_slice(payload);
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
        gate.validate()
            .unwrap_or_else(|err| panic!("Invalid gate command: {err}"));
        assert!(
            !gate.is_channel(),
            "Channel gates carry typed payloads and cannot be encoded in ByteMessage gate commands"
        );
        self.add_gate_parts_from_usizes(
            gate.gate_type,
            gate.qubits.len(),
            gate.qubits.iter().map(|qubit| usize::from(*qubit)),
            &gate.angles,
            &gate.params,
        )
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
    pub fn idle(&mut self, duration: f64, qubits: &[usize]) -> &mut Self {
        if qubits.is_empty() {
            return self;
        }

        if qubits.len() == 1 {
            return self.add_single_qubit_gate_parts(GateType::Idle, qubits[0], &[], &[duration]);
        }

        self.add_gate_parts(GateType::Idle, qubits, &[], &[duration])
    }

    /// Add an X gate
    pub fn x(&mut self, qubits: &[usize]) -> &mut Self {
        if qubits.len() == 1 {
            return self.add_single_qubit_gate_parts(GateType::X, qubits[0], &[], &[]);
        }
        self.add_gate_parts(GateType::X, qubits, &[], &[])
    }

    /// Add a Y gate
    pub fn y(&mut self, qubits: &[usize]) -> &mut Self {
        if qubits.len() == 1 {
            return self.add_single_qubit_gate_parts(GateType::Y, qubits[0], &[], &[]);
        }
        self.add_gate_parts(GateType::Y, qubits, &[], &[])
    }

    /// Add a Z gate
    pub fn z(&mut self, qubits: &[usize]) -> &mut Self {
        if qubits.len() == 1 {
            return self.add_single_qubit_gate_parts(GateType::Z, qubits[0], &[], &[]);
        }
        self.add_gate_parts(GateType::Z, qubits, &[], &[])
    }

    /// Add an H gate
    pub fn h(&mut self, qubits: &[usize]) -> &mut Self {
        if qubits.len() == 1 {
            return self.add_single_qubit_gate_parts(GateType::H, qubits[0], &[], &[]);
        }
        self.add_gate_parts(GateType::H, qubits, &[], &[])
    }

    /// Add CX (controlled-X) gates between pairs of qubits.
    ///
    /// Each tuple is a (control, target) pair.
    pub fn cx(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        if let [(control, target)] = pairs {
            return self.add_two_qubit_gate_parts(GateType::CX, *control, *target, &[], &[]);
        }
        self.add_gate_parts_from_usizes(
            GateType::CX,
            pairs.len() * 2,
            pairs
                .iter()
                .copied()
                .flat_map(|(control, target)| [control, target]),
            &[],
            &[],
        )
    }

    /// Add RZZ gates between pairs of qubits.
    ///
    /// Each tuple is a (qubit1, qubit2) pair.
    pub fn rzz(&mut self, theta: Angle64, pairs: &[(usize, usize)]) -> &mut Self {
        if let [(qubit1, qubit2)] = pairs {
            return self.add_two_qubit_gate_parts(GateType::RZZ, *qubit1, *qubit2, &[theta], &[]);
        }
        self.add_gate_parts_from_usizes(
            GateType::RZZ,
            pairs.len() * 2,
            pairs
                .iter()
                .copied()
                .flat_map(|(qubit1, qubit2)| [qubit1, qubit2]),
            &[theta],
            &[],
        )
    }

    /// Add SZZ gates between pairs of qubits.
    ///
    /// Each tuple is a (qubit1, qubit2) pair.
    pub fn szz(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        if let [(qubit1, qubit2)] = pairs {
            return self.add_two_qubit_gate_parts(GateType::SZZ, *qubit1, *qubit2, &[], &[]);
        }
        self.add_gate_parts_from_usizes(
            GateType::SZZ,
            pairs.len() * 2,
            pairs
                .iter()
                .copied()
                .flat_map(|(qubit1, qubit2)| [qubit1, qubit2]),
            &[],
            &[],
        )
    }

    /// Add `SZZdg` gates between pairs of qubits.
    ///
    /// Each tuple is a (qubit1, qubit2) pair.
    pub fn szzdg(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        if let [(qubit1, qubit2)] = pairs {
            return self.add_two_qubit_gate_parts(GateType::SZZdg, *qubit1, *qubit2, &[], &[]);
        }
        self.add_gate_parts_from_usizes(
            GateType::SZZdg,
            pairs.len() * 2,
            pairs
                .iter()
                .copied()
                .flat_map(|(qubit1, qubit2)| [qubit1, qubit2]),
            &[],
            &[],
        )
    }

    /// Add an RZ gate
    pub fn rz(&mut self, theta: Angle64, qubits: &[usize]) -> &mut Self {
        if qubits.len() == 1 {
            return self.add_single_qubit_gate_parts(GateType::RZ, qubits[0], &[theta], &[]);
        }
        self.add_gate_parts(GateType::RZ, qubits, &[theta], &[])
    }

    /// Add an R1XY gate
    pub fn r1xy(&mut self, theta: Angle64, phi: Angle64, qubits: &[usize]) -> &mut Self {
        if qubits.len() == 1 {
            return self.add_single_qubit_gate_parts(GateType::R1XY, qubits[0], &[theta, phi], &[]);
        }
        self.add_gate_parts(GateType::R1XY, qubits, &[theta, phi], &[])
    }

    /// Add a U gate
    pub fn u(
        &mut self,
        theta: Angle64,
        phi: Angle64,
        lambda: Angle64,
        qubits: &[usize],
    ) -> &mut Self {
        if qubits.len() == 1 {
            return self.add_single_qubit_gate_parts(
                GateType::U,
                qubits[0],
                &[theta, phi, lambda],
                &[],
            );
        }
        self.add_gate_parts(GateType::U, qubits, &[theta, phi, lambda], &[])
    }

    /// Add measurement operations for multiple qubits
    ///
    /// # Panics
    ///
    /// Panics if any qubit ID is too large to fit in a u32.
    pub fn mz(&mut self, qubit_ids: &[usize]) -> &mut Self {
        for &qubit in qubit_ids {
            self.add_single_qubit_gate_parts(GateType::MZ, qubit, &[], &[]);
        }
        self
    }

    /// Add measure leakage operations for multiple qubits
    ///
    /// This behaves like `mz()` but is intended for measuring qubits
    /// that may be in a leaked state. In the future, this will output 0, 1, or 2
    /// (where 2 indicates the qubit is leaked).
    ///
    /// # Panics
    ///
    /// Panics if any qubit ID is too large to fit in a u32.
    pub fn measure_leakages(&mut self, qubit_ids: &[usize]) -> &mut Self {
        for &qubit in qubit_ids {
            self.add_single_qubit_gate_parts(GateType::MeasureLeaked, qubit, &[], &[]);
        }
        self
    }

    /// Add a `MeasCrosstalkGlobalPayload`
    pub fn meas_crosstalk_global_payload(&mut self, qubits: &[usize]) -> &mut Self {
        self.add_gate_parts(GateType::MeasCrosstalkGlobalPayload, qubits, &[], &[])
    }

    /// Add a `MeasCrosstalkLocalPayload`
    pub fn meas_crosstalk_local_payload(&mut self, qubits: &[usize]) -> &mut Self {
        self.add_gate_parts(GateType::MeasCrosstalkLocalPayload, qubits, &[], &[])
    }

    /// Add a PZ (preparation/reset) gate
    pub fn pz(&mut self, qubits: &[usize]) -> &mut Self {
        if qubits.len() == 1 {
            return self.add_single_qubit_gate_parts(GateType::PZ, qubits[0], &[], &[]);
        }
        self.add_gate_parts(GateType::PZ, qubits, &[], &[])
    }

    /// Add an SZ (S) gate
    pub fn sz(&mut self, qubits: &[usize]) -> &mut Self {
        // S gate is RZ(π/2)
        self.rz(Angle64::QUARTER_TURN, qubits)
    }

    /// Add an `SZdg` (S†) gate
    pub fn szdg(&mut self, qubits: &[usize]) -> &mut Self {
        // S† gate is RZ(-π/2)
        self.rz(-Angle64::QUARTER_TURN, qubits)
    }

    /// Add a T gate
    pub fn t(&mut self, qubits: &[usize]) -> &mut Self {
        // T gate is RZ(π/4)
        self.rz(Angle64::QUARTER_TURN / 2u64, qubits)
    }

    /// Add a Tdg (T†) gate
    pub fn tdg(&mut self, qubits: &[usize]) -> &mut Self {
        // T† gate is RZ(-π/4)
        self.rz(-(Angle64::QUARTER_TURN / 2u64), qubits)
    }

    /// Add an RX gate
    pub fn rx(&mut self, theta: Angle64, qubits: &[usize]) -> &mut Self {
        if qubits.len() == 1 {
            return self.add_single_qubit_gate_parts(GateType::RX, qubits[0], &[theta], &[]);
        }
        self.add_gate_parts(GateType::RX, qubits, &[theta], &[])
    }

    /// Add an RY gate
    pub fn ry(&mut self, theta: Angle64, qubits: &[usize]) -> &mut Self {
        if qubits.len() == 1 {
            return self.add_single_qubit_gate_parts(GateType::RY, qubits[0], &[theta], &[]);
        }
        self.add_gate_parts(GateType::RY, qubits, &[theta], &[])
    }

    /// Add CY gates between pairs of qubits.
    ///
    /// Each tuple is a (control, target) pair.
    pub fn cy(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        if let [(control, target)] = pairs {
            return self.add_two_qubit_gate_parts(GateType::CY, *control, *target, &[], &[]);
        }
        self.add_gate_parts_from_usizes(
            GateType::CY,
            pairs.len() * 2,
            pairs
                .iter()
                .copied()
                .flat_map(|(control, target)| [control, target]),
            &[],
            &[],
        )
    }

    /// Add CZ gates between pairs of qubits.
    ///
    /// Each tuple is a (control, target) pair.
    pub fn cz(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        if let [(control, target)] = pairs {
            return self.add_two_qubit_gate_parts(GateType::CZ, *control, *target, &[], &[]);
        }
        self.add_gate_parts_from_usizes(
            GateType::CZ,
            pairs.len() * 2,
            pairs
                .iter()
                .copied()
                .flat_map(|(control, target)| [control, target]),
            &[],
            &[],
        )
    }

    /// Add an SX (sqrt-X) gate
    pub fn sx(&mut self, qubits: &[usize]) -> &mut Self {
        self.add_gate_parts(GateType::SX, qubits, &[], &[])
    }

    /// Add an `SXdg` (sqrt-X dagger) gate
    pub fn sxdg(&mut self, qubits: &[usize]) -> &mut Self {
        self.add_gate_parts(GateType::SXdg, qubits, &[], &[])
    }

    /// Add an SY (sqrt-Y) gate
    pub fn sy(&mut self, qubits: &[usize]) -> &mut Self {
        self.add_gate_parts(GateType::SY, qubits, &[], &[])
    }

    /// Add an `SYdg` (sqrt-Y dagger) gate
    pub fn sydg(&mut self, qubits: &[usize]) -> &mut Self {
        self.add_gate_parts(GateType::SYdg, qubits, &[], &[])
    }

    /// Add SWAP gates between pairs of qubits.
    ///
    /// Each tuple is a (qubit1, qubit2) pair.
    pub fn swap(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        if let [(qubit1, qubit2)] = pairs {
            return self.add_two_qubit_gate_parts(GateType::SWAP, *qubit1, *qubit2, &[], &[]);
        }
        self.add_gate_parts_from_usizes(
            GateType::SWAP,
            pairs.len() * 2,
            pairs
                .iter()
                .copied()
                .flat_map(|(qubit1, qubit2)| [qubit1, qubit2]),
            &[],
            &[],
        )
    }

    /// Add SXX gates between pairs of qubits.
    ///
    /// Each tuple is a (qubit1, qubit2) pair.
    pub fn sxx(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        if let [(qubit1, qubit2)] = pairs {
            return self.add_two_qubit_gate_parts(GateType::SXX, *qubit1, *qubit2, &[], &[]);
        }
        self.add_gate_parts_from_usizes(
            GateType::SXX,
            pairs.len() * 2,
            pairs
                .iter()
                .copied()
                .flat_map(|(qubit1, qubit2)| [qubit1, qubit2]),
            &[],
            &[],
        )
    }

    /// Add `SXXdg` gates between pairs of qubits.
    ///
    /// Each tuple is a (qubit1, qubit2) pair.
    pub fn sxxdg(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        if let [(qubit1, qubit2)] = pairs {
            return self.add_two_qubit_gate_parts(GateType::SXXdg, *qubit1, *qubit2, &[], &[]);
        }
        self.add_gate_parts_from_usizes(
            GateType::SXXdg,
            pairs.len() * 2,
            pairs
                .iter()
                .copied()
                .flat_map(|(qubit1, qubit2)| [qubit1, qubit2]),
            &[],
            &[],
        )
    }

    /// Add SYY gates between pairs of qubits.
    ///
    /// Each tuple is a (qubit1, qubit2) pair.
    pub fn syy(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        if let [(qubit1, qubit2)] = pairs {
            return self.add_two_qubit_gate_parts(GateType::SYY, *qubit1, *qubit2, &[], &[]);
        }
        self.add_gate_parts_from_usizes(
            GateType::SYY,
            pairs.len() * 2,
            pairs
                .iter()
                .copied()
                .flat_map(|(qubit1, qubit2)| [qubit1, qubit2]),
            &[],
            &[],
        )
    }

    /// Add `SYYdg` gates between pairs of qubits.
    ///
    /// Each tuple is a (qubit1, qubit2) pair.
    pub fn syydg(&mut self, pairs: &[(usize, usize)]) -> &mut Self {
        if let [(qubit1, qubit2)] = pairs {
            return self.add_two_qubit_gate_parts(GateType::SYYdg, *qubit1, *qubit2, &[], &[]);
        }
        self.add_gate_parts_from_usizes(
            GateType::SYYdg,
            pairs.len() * 2,
            pairs
                .iter()
                .copied()
                .flat_map(|(qubit1, qubit2)| [qubit1, qubit2]),
            &[],
            &[],
        )
    }

    /// Add RXX gates between pairs of qubits.
    ///
    /// Each tuple is a (qubit1, qubit2) pair.
    pub fn rxx(&mut self, theta: Angle64, pairs: &[(usize, usize)]) -> &mut Self {
        if let [(qubit1, qubit2)] = pairs {
            return self.add_two_qubit_gate_parts(GateType::RXX, *qubit1, *qubit2, &[theta], &[]);
        }
        self.add_gate_parts_from_usizes(
            GateType::RXX,
            pairs.len() * 2,
            pairs
                .iter()
                .copied()
                .flat_map(|(qubit1, qubit2)| [qubit1, qubit2]),
            &[theta],
            &[],
        )
    }

    /// Add RYY gates between pairs of qubits.
    ///
    /// Each tuple is a (qubit1, qubit2) pair.
    pub fn ryy(&mut self, theta: Angle64, pairs: &[(usize, usize)]) -> &mut Self {
        if let [(qubit1, qubit2)] = pairs {
            return self.add_two_qubit_gate_parts(GateType::RYY, *qubit1, *qubit2, &[theta], &[]);
        }
        self.add_gate_parts_from_usizes(
            GateType::RYY,
            pairs.len() * 2,
            pairs
                .iter()
                .copied()
                .flat_map(|(qubit1, qubit2)| [qubit1, qubit2]),
            &[theta],
            &[],
        )
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
    /// builder.h(&[0]);
    /// let message1 = builder.build();
    ///
    /// // Reset and configure for next message
    /// builder.reset();
    /// let _ = builder.for_quantum_operations();
    /// builder.h(&[1]);
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
    #[should_panic(
        expected = "Channel gates carry typed payloads and cannot be encoded in ByteMessage gate commands"
    )]
    fn test_add_gate_command_rejects_channel_gate() {
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        let gate = Gate::channel(pecos_core::channel::Depolarizing(0.01, 0));
        builder.add_gate_command(&gate);
    }

    #[test]
    #[should_panic(expected = "Invalid gate command")]
    fn test_add_gate_command_rejects_invalid_gate_payload() {
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        let gate = Gate::cx(&[(0, 0)]);
        builder.add_gate_command(&gate);
    }

    #[test]
    fn test_builder_basic() {
        // Create a builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Add some gates
        builder.h(&[0]);
        builder.cx(&[(0, 1)]);
        builder.mz(&[2]);

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
        assert_eq!(commands[2].gate_type, GateType::MZ);
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
        builder.h(&[0]);
        builder.x(&[1]);
        builder.y(&[2]);
        builder.z(&[3]);
        builder.rz(Angle64::from_radians(0.5), &[4]);
        builder.r1xy(Angle64::from_radians(0.1), Angle64::from_radians(0.2), &[5]);
        builder.mz(&[6]);

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
        assert_eq!(commands[6].gate_type, GateType::MZ);
    }

    #[test]
    fn test_single_item_fast_paths_match_generic_gate_encoding() {
        let theta_rx = Angle64::from_radians(0.3);
        let theta_ry = Angle64::from_radians(0.4);
        let theta_rz = Angle64::from_radians(0.5);
        let theta_r1xy = Angle64::from_radians(0.6);
        let phi_r1xy = Angle64::from_radians(0.7);
        let theta_rzz = Angle64::from_radians(0.8);

        let mut generic_builder = ByteMessageBuilder::new();
        let _ = generic_builder.for_quantum_operations();
        generic_builder.add_gate_command(&Gate::rx(theta_rx, &[1]));
        generic_builder.add_gate_command(&Gate::ry(theta_ry, &[2]));
        generic_builder.add_gate_command(&Gate::rz(theta_rz, &[3]));
        generic_builder.add_gate_command(&Gate::r1xy(theta_r1xy, phi_r1xy, &[4]));
        generic_builder.add_gate_command(&Gate::rzz(theta_rzz, &[(5, 6)]));

        let mut fast_path_builder = ByteMessageBuilder::new();
        let _ = fast_path_builder.for_quantum_operations();
        fast_path_builder.rx(theta_rx, &[1]);
        fast_path_builder.ry(theta_ry, &[2]);
        fast_path_builder.rz(theta_rz, &[3]);
        fast_path_builder.r1xy(theta_r1xy, phi_r1xy, &[4]);
        fast_path_builder.rzz(theta_rzz, &[(5, 6)]);

        let generic_message = generic_builder.build();
        let fast_path_message = fast_path_builder.build();

        assert_eq!(generic_message.as_bytes(), fast_path_message.as_bytes());
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
        builder.h(&[0]);
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
        builder.mz(&qubits);

        // Build the message
        let message = builder.build();

        // Parse the message
        let commands = message.quantum_ops().unwrap();

        // Verify the commands
        assert_eq!(commands.len(), 3);
        for i in 0..3 {
            assert_eq!(commands[i].gate_type, GateType::MZ);
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
        builder.measure_leakages(&qubits);

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
        builder.h(&[0]);

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
        builder.h(&[0]);

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
        builder.h(&[0]);
        builder.cx(&[(0, 1)]);

        // Check the message count
        assert_eq!(builder.message_count(), 2);

        // Clear the builder
        builder.clear();

        // Check the message count after clearing
        assert_eq!(builder.message_count(), 0);

        // Add a new gate
        builder.h(&[0]);

        // Check the message count again
        assert_eq!(builder.message_count(), 1);
    }

    #[test]
    fn test_reset() {
        // Create a builder
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Add some gates
        builder.h(&[0]);
        builder.cx(&[(0, 1)]);

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
        builder.h(&[0]);

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
                    builder.h(&[0]);

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
                    builder.h(&[0]);

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
