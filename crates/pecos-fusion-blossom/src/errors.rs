//! Error types for Fusion Blossom decoder

use thiserror::Error;

/// Error type for Fusion Blossom operations
#[derive(Error, Debug)]
pub enum FusionBlossomError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Invalid graph structure
    #[error("Invalid graph: {0}")]
    InvalidGraph(String),

    /// Decoding failed
    #[error("Decoding failed: {0}")]
    DecodingFailed(String),

    /// Invalid syndrome pattern
    #[error("Invalid syndrome pattern: {0}")]
    InvalidSyndrome(String),

    /// Invalid check matrix
    #[error("Invalid check matrix: {0}")]
    InvalidCheckMatrix(String),
}

/// Result type for Fusion Blossom operations
pub type Result<T> = std::result::Result<T, FusionBlossomError>;

/// Convert `FusionBlossomError` to `DecoderError`
impl From<FusionBlossomError> for pecos_decoder_core::DecoderError {
    fn from(e: FusionBlossomError) -> Self {
        match e {
            FusionBlossomError::Configuration(msg) => {
                pecos_decoder_core::DecoderError::InvalidConfiguration(msg)
            }
            FusionBlossomError::InvalidGraph(msg) => {
                pecos_decoder_core::DecoderError::InvalidGraph(msg)
            }
            FusionBlossomError::DecodingFailed(msg) => {
                pecos_decoder_core::DecoderError::DecodingFailed(msg)
            }
            FusionBlossomError::InvalidSyndrome(msg) => {
                pecos_decoder_core::DecoderError::InvalidSyndrome(msg)
            }
            FusionBlossomError::InvalidCheckMatrix(msg) => {
                pecos_decoder_core::DecoderError::MatrixError(msg)
            }
        }
    }
}
