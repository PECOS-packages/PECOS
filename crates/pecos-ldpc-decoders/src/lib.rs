//! LDPC (Low-Density Parity-Check) decoder implementations for PECOS
//!
//! This crate provides various LDPC decoder implementations including:
//! - Belief Propagation with Ordered Statistics Decoding (BP+OSD)
//! - Belief Propagation with Localised Statistics Decoding (BP+LSD)
//! - Soft Information BP decoder
//! - Bit-flipping decoder
//! - Union-Find decoder
//! - `BeliefFind` decoder (BP + Union-Find hybrid)
//! - MBP decoder for quantum codes

use ndarray::Array1;
use std::os::raw::c_int;
use thiserror::Error;

// Internal modules
mod bridge;
pub mod builders;
pub mod core_traits_simple;
pub mod decoders;
pub mod quantum;
pub mod sparse;

// Re-export main decoder types
pub use decoders::{
    BeliefFindDecoder, BpLsdDecoder, BpOsdDecoder, ClusterStatistics, FlipDecoder, LsdStatistics,
    SoftInfoBpDecoder, UfMethod, UnionFindDecoder,
};
pub use quantum::{CssCode, MbpDecoder};
pub use sparse::SparseMatrix;

// Re-export builders
pub use builders::{
    BeliefFindBuilder, BpLsdBuilder, BpOsdBuilder, DecoderBuilder, FlipBuilder, SoftInfoBpBuilder,
    UnionFindBuilder,
};

/// Error type for LDPC decoder operations
#[derive(Error, Debug)]
pub enum LdpcError {
    #[error("Invalid input dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: usize, actual: usize },

    #[error("Invalid parity check matrix")]
    InvalidMatrix(String),

    #[error("Decoder failed to converge after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("FFI error: {0}")]
    FfiError(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("LDPC error: {0}")]
    Ldpc(String),
}

/// Result type alias for LDPC operations
pub type Result<T> = std::result::Result<T, LdpcError>;

/// Belief Propagation decoding method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpMethod {
    /// Product-sum algorithm
    ProductSum,
    /// Minimum-sum algorithm
    MinimumSum,
}

impl BpMethod {
    pub(crate) fn to_ffi(self) -> c_int {
        match self {
            BpMethod::ProductSum => 0,
            BpMethod::MinimumSum => 1,
        }
    }
}

/// Belief Propagation scheduling method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BpSchedule {
    /// Serial schedule
    Serial,
    /// Parallel schedule
    Parallel,
    /// Serial relative schedule
    SerialRelative,
}

impl BpSchedule {
    pub(crate) fn to_ffi(self) -> c_int {
        match self {
            BpSchedule::Serial => 0,
            BpSchedule::Parallel => 1,
            BpSchedule::SerialRelative => 2,
        }
    }
}

/// OSD (Ordered Statistics Decoding) method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OsdMethod {
    /// OSD disabled
    Off,
    /// OSD-0 (order 0)
    Osd0,
    /// OSD-E (exhaustive)
    OsdE,
    /// OSD-CS (combination sweep)
    OsdCs,
}

impl OsdMethod {
    pub(crate) fn to_ffi(self) -> c_int {
        match self {
            OsdMethod::Off => 0,
            OsdMethod::Osd0 => 1,
            OsdMethod::OsdE => 2,
            OsdMethod::OsdCs => 3,
        }
    }
}

/// Input vector type for decoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputVectorType {
    /// Syndrome vector (length = number of checks)
    Syndrome,
    /// Received vector (length = number of bits)
    ReceivedVector,
    /// Automatically detect based on input size
    Auto,
}

impl InputVectorType {
    pub(crate) fn to_ffi(self) -> c_int {
        match self {
            InputVectorType::Syndrome => 0,
            InputVectorType::ReceivedVector => 1,
            InputVectorType::Auto => 2,
        }
    }
}

/// Result of a decoding operation
#[derive(Debug, Clone)]
pub struct DecodingResult {
    /// The decoded error vector
    pub decoding: Array1<u8>,
    /// Whether the decoder converged
    pub converged: bool,
    /// Number of iterations performed
    pub iterations: usize,
}

// DecodingResultTrait implementation moved to core_traits.rs

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bp_method_conversion() {
        assert_eq!(BpMethod::ProductSum.to_ffi(), 0);
        assert_eq!(BpMethod::MinimumSum.to_ffi(), 1);
    }

    #[test]
    fn test_bp_schedule_conversion() {
        assert_eq!(BpSchedule::Serial.to_ffi(), 0);
        assert_eq!(BpSchedule::Parallel.to_ffi(), 1);
        assert_eq!(BpSchedule::SerialRelative.to_ffi(), 2);
    }
}
