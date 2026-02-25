//! Error types for Relay BP decoder

use thiserror::Error;

/// Error type for Relay BP operations
#[derive(Error, Debug)]
pub enum RelayBpError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Invalid check matrix
    #[error("Invalid matrix: {0}")]
    InvalidMatrix(String),

    /// Decoding failed
    #[error("Decoding failed: {0}")]
    DecodingFailed(String),

    /// Invalid syndrome pattern
    #[error("Invalid syndrome: {0}")]
    InvalidSyndrome(String),
}

/// Result type for Relay BP operations
pub type Result<T> = std::result::Result<T, RelayBpError>;

/// Convert `RelayBpError` to `DecoderError`
impl From<RelayBpError> for pecos_decoder_core::DecoderError {
    fn from(e: RelayBpError) -> Self {
        match e {
            RelayBpError::Configuration(msg) => {
                pecos_decoder_core::DecoderError::InvalidConfiguration(msg)
            }
            RelayBpError::InvalidMatrix(msg) => pecos_decoder_core::DecoderError::MatrixError(msg),
            RelayBpError::DecodingFailed(msg) => {
                pecos_decoder_core::DecoderError::DecodingFailed(msg)
            }
            RelayBpError::InvalidSyndrome(msg) => {
                pecos_decoder_core::DecoderError::InvalidSyndrome(msg)
            }
        }
    }
}
