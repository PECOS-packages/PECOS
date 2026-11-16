//! Protocol definitions for the byte-level messaging system
//!
//! This module defines the message formats, headers, and constants
//! used in the byte protocol.

use bytemuck::{Pod, Zeroable};

// Magic bytes to identify PECOS message batches - "PECS"
pub const BATCH_MAGIC: u32 = 0x50_45_43_53;
// Version 8 widens ordinary gate qubit counts to u32, retaining version 5's
// u64 angle fractions. Versions 6/7 encode scheduled batches/events; version 2
// is the runtime envelope.
// Legacy versions 1 (ordinary) and 3/4 (scheduled) used lossy f64 radians.
pub const PROTOCOL_VERSION: u8 = 8;
/// Scheduled frame transport with exact angle fractions.
pub const SCHEDULED_FRAME_VERSION: u8 = 6;
/// Scheduled event transport with exact angle fractions.
pub const SCHEDULED_EVENT_VERSION: u8 = 7;

/// Flags that can be set on individual messages.
///
/// No flag is defined yet: all eight bits of the header's flag byte are
/// reserved for future use, and [`MessageHeader::get_flags`] returns them as
/// carried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageFlags(u8);

impl MessageFlags {
    /// No flags set.
    pub const NONE: Self = Self(0);

    /// The flag byte as written to the message header.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }
}

/// Message types used in the protocol
#[repr(u8)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MessageType {
    // Operation messages
    Gate = 10, // All gate operations (including measurements)

    // Result messages
    Outcome = 20,     // Measurement result
    ReturnValue = 21, // Program return value (from teardown or main function)
}

/// Message batch header for framing multiple messages
#[repr(C, align(4))]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct BatchHeader {
    pub magic: u32,      // Magic number 'PEQS'
    pub version: u8,     // Protocol version
    pub flags: u8,       // Batch flags
    pub reserved: u16,   // Reserved for future use (padding for alignment)
    pub msg_count: u32,  // Number of messages in batch
    pub total_size: u32, // Total size in bytes including this header
}

impl BatchHeader {
    /// Create a new batch header
    #[must_use]
    pub fn new(msg_count: u32, total_size: u32) -> Self {
        Self {
            magic: BATCH_MAGIC,
            version: PROTOCOL_VERSION,
            flags: 0,
            reserved: 0,
            msg_count,
            total_size,
        }
    }

    /// Check if the header has a valid magic number and version
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.magic == BATCH_MAGIC && self.version == PROTOCOL_VERSION
    }
}

/// Individual message header
#[repr(C, align(4))]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct MessageHeader {
    pub msg_type: u8,      // Message type
    pub flags: u8,         // Message flags
    pub reserved: u16,     // Reserved for future use (padding for alignment)
    pub payload_size: u32, // Size of payload following this header
}

impl MessageHeader {
    /// Create a new message header
    #[must_use]
    pub fn new(msg_type: MessageType, payload_size: u32, flags: MessageFlags) -> Self {
        Self {
            msg_type: msg_type as u8,
            flags: flags.bits(),
            reserved: 0,
            payload_size,
        }
    }

    /// Get the message type from a raw header
    ///
    /// # Errors
    ///
    /// Returns an error if the message type is unknown or invalid.
    pub fn get_type(&self) -> Result<MessageType, &'static str> {
        match self.msg_type {
            10 => Ok(MessageType::Gate),
            20 => Ok(MessageType::Outcome),
            21 => Ok(MessageType::ReturnValue),
            _ => Err("Unknown message type"),
        }
    }

    /// Get the message flags from a raw header
    #[must_use]
    pub fn get_flags(&self) -> MessageFlags {
        MessageFlags(self.flags)
    }
}

/// Gate message payload header
#[repr(C, align(4))]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct GateHeader {
    pub gate_type: u8,  // Gate type (using GateType enum values)
    pub has_params: u8, // Whether gate has parameters (1=yes, 0=no)
    pub reserved: u16,  // Reserved for future use
    pub num_qubits: u32, // Number of qubits
                        // Followed by:
                        // - qubit_indices: [u32; num_qubits]
                        // - angles: little-endian u64 Angle64 fractions (angle_arity words)
                        // - non-angle parameters: little-endian f64 words
                        // Custom treats every parameter word as an angle.
}

// MeasurementHeader removed - measurements now use regular gate structure

/// Measurement result message payload header
#[repr(C, align(4))]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct OutcomeHeader {
    pub outcome: u32, // Measurement outcome (0 or 1, but u32 for alignment)
}

/// Return value message payload header
#[repr(C, align(4))]
#[derive(Debug, Copy, Clone, Pod, Zeroable)]
pub struct ReturnValueHeader {
    pub value: i64, // Return value from program (i64 for general integer support)
}

/// Calculate padding needed for alignment
#[must_use]
pub fn calc_padding(offset: usize, alignment: usize) -> usize {
    (alignment - (offset % alignment)) % alignment
}
