// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Error types for the MWPF decoder

use thiserror::Error;

/// Error type for MWPF operations
#[derive(Error, Debug)]
pub enum MwpfError {
    /// Configuration error
    #[error("Configuration error: {0}")]
    Configuration(String),

    /// Decoding failed
    #[error("Decoding failed: {0}")]
    DecodingFailed(String),

    /// Invalid DEM format
    #[error("Invalid DEM: {0}")]
    InvalidDem(String),

    /// Syndrome the model cannot produce
    #[error("Invalid syndrome: {0}")]
    InvalidSyndrome(String),

    /// Syndrome of the wrong length
    #[error("Invalid syndrome length: expected {expected} detectors, got {actual}")]
    InvalidDimensions {
        /// Number of detectors in the model
        expected: usize,
        /// Length of the syndrome given
        actual: usize,
    },
}

/// Result type for MWPF operations
pub type Result<T> = std::result::Result<T, MwpfError>;

/// Convert `MwpfError` to `DecoderError`
impl From<MwpfError> for pecos_decoder_core::DecoderError {
    fn from(e: MwpfError) -> Self {
        match e {
            MwpfError::Configuration(msg) | MwpfError::InvalidDem(msg) => {
                pecos_decoder_core::DecoderError::InvalidConfiguration(msg)
            }
            MwpfError::DecodingFailed(msg) => pecos_decoder_core::DecoderError::DecodingFailed(msg),
            MwpfError::InvalidSyndrome(msg) => {
                pecos_decoder_core::DecoderError::InvalidSyndrome(msg)
            }
            MwpfError::InvalidDimensions { expected, actual } => {
                pecos_decoder_core::DecoderError::InvalidDimensions { expected, actual }
            }
        }
    }
}
