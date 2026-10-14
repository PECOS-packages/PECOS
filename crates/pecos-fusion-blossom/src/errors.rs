//! Error types for Fusion Blossom decoder

use thiserror::Error;

/// Error type for Fusion Blossom operations
#[derive(Error, Debug)]
pub enum FusionBlossomError {
    /// Malformed DEM text, preserving the shared grammar error.
    #[error("{0}")]
    DemSyntax(#[source] pecos_decoder_core::DecoderError),
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

impl From<pecos_decoder_core::DecoderError> for FusionBlossomError {
    fn from(error: pecos_decoder_core::DecoderError) -> Self {
        match error {
            pecos_decoder_core::DecoderError::InvalidDemSyntax(_) => Self::DemSyntax(error),
            error => Self::Configuration(error.to_string()),
        }
    }
}

/// Convert `FusionBlossomError` to `DecoderError`
impl From<FusionBlossomError> for pecos_decoder_core::DecoderError {
    fn from(e: FusionBlossomError) -> Self {
        match e {
            FusionBlossomError::DemSyntax(error) => error,
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
