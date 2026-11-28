//! Improved error types for `PyMatching` decoder

use pecos_decoder_core::DecoderError;
use thiserror::Error;

/// Specific error types for `PyMatching` operations
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PyMatchingError {
    /// Shared validation failed (DEM text or batch buffer size)
    #[error("{0}")]
    Dem(#[from] DecoderError),

    /// FFI-related errors from the C++ library
    #[error("FFI error: {0}")]
    Ffi(cxx::Exception),

    /// Invalid check matrix
    #[error("Invalid check matrix: {0}")]
    InvalidCheckMatrix(CheckMatrixError),

    /// Invalid syndrome
    #[error("Invalid syndrome: expected length {expected}, got {actual}")]
    InvalidSyndrome { expected: usize, actual: usize },

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Upstream `PyMatching` or Stim failed in a graph-phase step, a rejection of
    /// the graph: parsing or building from DEM text (including its edge weights),
    /// creating a graph, adding edges, setting the boundary, or building the
    /// matcher. Any C++ exception there counts, including `std::bad_alloc` and
    /// `std::length_error`, since the graph's size comes from caller values. A
    /// decode-phase failure, such as an odd-parity syndrome with no perfect
    /// matching, stays [`PyMatchingError::Ffi`]. Holds the upstream text.
    #[error("{0}")]
    InvalidGraph(String),

    /// File I/O error
    #[error("File I/O error: {0}")]
    FileIo(#[from] std::io::Error),
}

/// Specific errors for check matrix operations
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum CheckMatrixError {
    #[error("All rows must have the same number of columns")]
    InconsistentColumns,

    #[error("Empty check matrix")]
    EmptyMatrix,
}

impl From<cxx::Exception> for PyMatchingError {
    /// A C++ exception the bridge marked as a graph-phase failure becomes
    /// [`PyMatchingError::InvalidGraph`] with the upstream text; any other stays
    /// [`PyMatchingError::Ffi`].
    fn from(error: cxx::Exception) -> Self {
        match error.what().strip_prefix(crate::bridge::INPUT_ERROR_PREFIX) {
            Some(message) => PyMatchingError::InvalidGraph(message.to_string()),
            None => PyMatchingError::Ffi(error),
        }
    }
}

impl From<PyMatchingError> for DecoderError {
    fn from(e: PyMatchingError) -> Self {
        match e {
            PyMatchingError::Dem(error) => error,
            PyMatchingError::Configuration(msg) => DecoderError::InvalidConfiguration(msg),
            PyMatchingError::InvalidGraph(msg) => DecoderError::InvalidGraph(msg),
            PyMatchingError::InvalidCheckMatrix(check_err) => {
                DecoderError::MatrixError(check_err.to_string())
            }
            PyMatchingError::InvalidSyndrome { expected, actual } => {
                DecoderError::InvalidDimensions { expected, actual }
            }
            PyMatchingError::Ffi(cxx_err) => DecoderError::FfiError(cxx_err.to_string()),
            PyMatchingError::FileIo(io_err) => DecoderError::IoError(io_err),
        }
    }
}

/// Result type for `PyMatching` operations
pub type Result<T> = std::result::Result<T, PyMatchingError>;
