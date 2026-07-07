//! Unified error types for PECOS decoders
//!
//! This module provides common error types that all decoder implementations
//! can use, ensuring consistent error handling across the ecosystem.

use thiserror::Error;

/// Common error type for all decoder operations
#[derive(Error, Debug)]
pub enum DecoderError {
    /// Invalid input dimensions
    #[error("Invalid input dimensions: expected {expected}, got {actual}")]
    InvalidDimensions { expected: usize, actual: usize },

    /// Decoder failed to converge
    #[error("Decoder failed to converge after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),

    /// Internal decoder error
    #[error("Internal decoder error: {0}")]
    InternalError(String),

    /// Invalid graph structure
    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),

    /// FFI error from C++ bindings
    #[error("FFI error: {0}")]
    FfiError(String),

    /// Matrix-related errors
    #[error("Matrix error: {0}")]
    MatrixError(String),

    /// Batch size mismatch
    #[error("Batch size mismatch: expected {expected}, got {actual}")]
    BatchSizeMismatch { expected: usize, actual: usize },

    /// Invalid syndrome
    #[error("Invalid syndrome: {0}")]
    InvalidSyndrome(String),

    /// Invalid node index
    #[error("Invalid node index {index}: must be < {max}")]
    InvalidNodeIndex { index: usize, max: usize },

    /// Observable-frame bit index out of range for the `u64` frame
    #[error(
        "Observable frame bit {bit} out of range: boundary-gate bits index a u64 frame and must be < 64"
    )]
    ObservableBitOutOfRange { bit: u32 },

    /// Invalid edge
    #[error("Invalid edge: {0}")]
    InvalidEdge(String),

    /// Decoding failure
    #[error("Decoding failed: {0}")]
    DecodingFailed(String),

    /// Not implemented
    #[error("Feature not implemented: {0}")]
    NotImplemented(String),

    /// I/O error
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Other errors (for compatibility)
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Specialized error for matrix operations
#[derive(Error, Debug)]
pub enum MatrixError {
    /// Invalid dimensions
    #[error("Invalid matrix dimensions: {rows}x{cols}")]
    InvalidDimensions { rows: usize, cols: usize },

    /// Dimension mismatch
    #[error(
        "Matrix dimension mismatch: expected {expected_rows}x{expected_cols}, got {actual_rows}x{actual_cols}"
    )]
    DimensionMismatch {
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },

    /// Empty matrix
    #[error("Matrix cannot be empty")]
    EmptyMatrix,

    /// Invalid index
    #[error("Matrix index out of bounds: ({row}, {col}) for matrix of size {rows}x{cols}")]
    IndexOutOfBounds {
        row: usize,
        col: usize,
        rows: usize,
        cols: usize,
    },

    /// Singular matrix
    #[error("Matrix is singular or near-singular")]
    SingularMatrix,

    /// Invalid format
    #[error("Invalid matrix format: {0}")]
    InvalidFormat(String),
}

/// Specialized error for graph operations
#[derive(Error, Debug)]
pub enum GraphError {
    /// Invalid node
    #[error("Invalid node {node}: must be < {num_nodes}")]
    InvalidNode { node: usize, num_nodes: usize },

    /// Invalid edge
    #[error("Invalid edge from {from} to {to}")]
    InvalidEdge { from: usize, to: usize },

    /// Disconnected graph
    #[error("Graph is disconnected")]
    DisconnectedGraph,

    /// No path exists
    #[error("No path exists from node {from} to node {to}")]
    NoPath { from: usize, to: usize },

    /// Duplicate edge
    #[error("Duplicate edge from {from} to {to}")]
    DuplicateEdge { from: usize, to: usize },

    /// Invalid weight
    #[error("Invalid edge weight: {weight}")]
    InvalidWeight { weight: f64 },
}

/// Specialized error for configuration
#[derive(Error, Debug)]
pub enum ConfigError {
    /// Missing required field
    #[error("Missing required configuration field: {0}")]
    MissingField(String),

    /// Invalid value
    #[error("Invalid value for {field}: {value}")]
    InvalidValue { field: String, value: String },

    /// Out of range
    #[error("Value {value} for {field} is out of range [{min}, {max}]")]
    OutOfRange {
        field: String,
        value: String,
        min: String,
        max: String,
    },

    /// Incompatible options
    #[error("Incompatible configuration options: {0}")]
    IncompatibleOptions(String),
}

/// Result type alias using `DecoderError`
pub type Result<T> = std::result::Result<T, DecoderError>;

/// Convert specialized errors to general `DecoderError`
impl From<MatrixError> for DecoderError {
    fn from(err: MatrixError) -> Self {
        DecoderError::MatrixError(err.to_string())
    }
}

impl From<GraphError> for DecoderError {
    fn from(err: GraphError) -> Self {
        DecoderError::InvalidGraph(err.to_string())
    }
}

impl From<ConfigError> for DecoderError {
    fn from(err: ConfigError) -> Self {
        DecoderError::InvalidConfiguration(err.to_string())
    }
}

/// Extension trait for converting between error types
pub trait ErrorConvert {
    /// Convert to `DecoderError`
    fn to_decoder_error(self) -> DecoderError;
}

impl<E: std::error::Error + Send + Sync + 'static> ErrorConvert for E {
    fn to_decoder_error(self) -> DecoderError {
        DecoderError::Other(anyhow::Error::new(self))
    }
}
