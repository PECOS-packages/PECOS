//! Core traits and utilities for PECOS decoders
//!
//! This crate defines the common traits and types that all decoder implementations
//! should use, enabling interoperability between different decoder types.
//!
//! # Structure
//!
//! - `errors` - Unified error types using thiserror
//! - `results` - Common result types and builders
//! - `config` - Configuration traits and validation utilities
//! - `matrix` - Common matrix types and check matrix traits
//! - `dem` - Detector error model traits and utilities

pub mod advanced;
pub mod config;
pub mod dem;
pub mod errors;
pub mod matrix;
pub mod results;

use ndarray::ArrayView1;

// Re-export commonly used types
pub use advanced::{
    AdvancedDecoder, AdvancedDecodingResult, DecodingOptions, DecodingStats, DetailedDecoder,
    DynamicWeightDecoder, ErasureDecoder, MatchedEdge, MatchedPair,
};
pub use config::{
    BatchConfig, ConfigBuilder, DecoderConfig, DecodingMethod, PerformanceConfig, SolverType,
};
pub use dem::{DemConfig, DemConfigBuilder, DemDecoder, DemInfo};
pub use errors::{ConfigError, DecoderError, ErrorConvert, GraphError, MatrixError};
pub use matrix::{CheckMatrixConfig, CheckMatrixDecoder, SparseCheckMatrix};
pub use results::{
    BatchDecodingResult, DecodingResultTrait, ResultBuilder, StandardDecodingResult,
};

/// Core trait that all decoders must implement
pub trait Decoder {
    /// The result type for this decoder
    type Result: DecodingResultTrait;

    /// The error type for this decoder
    type Error: std::error::Error + Send + Sync + 'static;

    /// Decode a syndrome or received vector
    ///
    /// The exact interpretation of the input depends on the decoder type
    /// and configuration. For LDPC decoders, this is typically a syndrome
    /// vector when using syndrome-based decoding.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The input dimensions don't match the decoder's expectations
    /// - The decoding process fails to converge
    /// - Internal decoder errors occur
    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error>;

    /// Get the number of checks (rows in parity check matrix)
    fn check_count(&self) -> usize;

    /// Get the number of bits (columns in parity check matrix)
    fn bit_count(&self) -> usize;
}

/// Trait for decoders that support soft information (log-likelihood ratios)
pub trait SoftDecoder: Decoder {
    /// Decode using soft information (log-likelihood ratios)
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The LLR array dimensions don't match the decoder's expectations
    /// - The LLR values are invalid (e.g., NaN or Inf)
    /// - The soft decoding process fails
    fn decode_soft(&mut self, llrs: &ArrayView1<f64>) -> Result<Self::Result, Self::Error>;
}

/// Trait for quantum CSS code decoders
pub trait CssDecoder {
    /// The result type for this decoder
    type Result: DecodingResultTrait;

    /// The error type for this decoder
    type Error: std::error::Error + Send + Sync + 'static;

    /// Decode both X and Z syndromes for a CSS code
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Either syndrome has incorrect dimensions
    /// - The X or Z decoding fails
    /// - The decoder doesn't support CSS decoding
    fn decode_css(
        &mut self,
        x_syndrome: &ArrayView1<u8>,
        z_syndrome: &ArrayView1<u8>,
    ) -> Result<Self::Result, Self::Error>;

    /// Get the number of X checks
    fn x_check_count(&self) -> usize;

    /// Get the number of Z checks
    fn z_check_count(&self) -> usize;

    /// Get the number of qubits
    fn qubit_count(&self) -> usize;
}

/// Trait for decoders that support batch decoding
pub trait BatchDecoder: Decoder {
    /// Decode multiple inputs in a batch
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Any input has incorrect dimensions
    /// - Any individual decoding fails
    /// - The batch is too large for the decoder to handle
    fn decode_batch(&mut self, inputs: &[ArrayView1<u8>])
    -> Result<Vec<Self::Result>, Self::Error>;
}

// ============================================================================
// Re-exports
// ============================================================================

/// Re-export common types
pub use ndarray;
pub use thiserror;
