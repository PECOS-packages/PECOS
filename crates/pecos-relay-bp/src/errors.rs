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

/// Preserve native messages without stacking error-category prefixes.
pub(crate) fn native_message(error: pecos_decoder_core::DecoderError) -> String {
    use pecos_decoder_core::DecoderError;
    match error {
        DecoderError::InvalidDemSyntax(message)
        | DecoderError::InvalidConfiguration(message)
        | DecoderError::InternalError(message)
        | DecoderError::InvalidGraph(message)
        | DecoderError::FfiError(message)
        | DecoderError::MatrixError(message)
        | DecoderError::InvalidSyndrome(message)
        | DecoderError::InvalidEdge(message)
        | DecoderError::DecodingFailed(message)
        | DecoderError::NotImplemented(message) => message,
        // Structured errors have no inner message to extract.
        error => error.to_string(),
    }
}
