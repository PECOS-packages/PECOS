//! Improved error types for `PyMatching` decoder

use pecos_decoder_core::DecoderError;
use thiserror::Error;

/// Specific error types for `PyMatching` operations
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum PyMatchingError {
    /// FFI-related errors from the C++ library
    #[error("FFI error: {0}")]
    Ffi(#[from] cxx::Exception),

    /// Invalid check matrix
    #[error("Invalid check matrix: {0}")]
    InvalidCheckMatrix(CheckMatrixError),

    /// Invalid syndrome
    #[error("Invalid syndrome: expected length {expected}, got {actual}")]
    InvalidSyndrome { expected: usize, actual: usize },

    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

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

impl From<PyMatchingError> for DecoderError {
    fn from(e: PyMatchingError) -> Self {
        match e {
            PyMatchingError::Configuration(msg) => DecoderError::InvalidConfiguration(msg),
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
