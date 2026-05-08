//! Simplified implementation of core decoder traits for LDPC decoders
//!
//! This module implements basic Decoder trait for LDPC decoder types.
//! The full `CheckMatrixDecoder` implementation is complex due to the many
//! parameters required by LDPC decoders.

use crate::decoders::{
    BeliefFindDecoder, BpLsdDecoder, BpOsdDecoder, FlipDecoder, SoftInfoBpDecoder, UnionFindDecoder,
};
use crate::{DecodingResult, LdpcError};
use ndarray::ArrayView1;
use pecos_decoder_core::{Decoder, DecodingResultTrait, StandardDecodingResult};

/// Convert `LdpcError` to `DecoderError`
impl From<LdpcError> for pecos_decoder_core::DecoderError {
    fn from(e: LdpcError) -> Self {
        match e {
            LdpcError::InvalidDimensions { expected, actual } => {
                pecos_decoder_core::DecoderError::InvalidDimensions { expected, actual }
            }
            LdpcError::InvalidMatrix(msg) => pecos_decoder_core::DecoderError::MatrixError(msg),
            LdpcError::ConvergenceFailure { iterations } => {
                pecos_decoder_core::DecoderError::ConvergenceFailure { iterations }
            }
            LdpcError::InvalidConfig(msg) | LdpcError::InvalidInput(msg) => {
                pecos_decoder_core::DecoderError::InvalidConfiguration(msg)
            }
            LdpcError::FfiError(msg) => pecos_decoder_core::DecoderError::FfiError(msg),
            LdpcError::Ldpc(msg) => pecos_decoder_core::DecoderError::InternalError(msg),
        }
    }
}

/// Update `DecodingResultTrait` implementation to include `to_standard`
impl DecodingResultTrait for DecodingResult {
    fn is_successful(&self) -> bool {
        self.converged
    }

    fn correction(&self) -> &[u8] {
        self.decoding.as_slice().unwrap_or(&[])
    }

    fn iterations(&self) -> Option<usize> {
        Some(self.iterations)
    }

    fn to_standard(&self) -> StandardDecodingResult {
        StandardDecodingResult {
            observable: self.decoding.to_vec(),
            weight: 0.0, // LDPC decoders don't typically provide weight
            converged: Some(self.converged),
            iterations: Some(self.iterations),
            confidence: None,
        }
    }
}

/// Implement Decoder trait for `BpOsdDecoder`
impl Decoder for BpOsdDecoder {
    type Result = DecodingResult;
    type Error = LdpcError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        self.decode(input)
    }

    fn check_count(&self) -> usize {
        self.check_count()
    }

    fn bit_count(&self) -> usize {
        self.bit_count()
    }
}

/// Implement Decoder trait for `BpLsdDecoder`
impl Decoder for BpLsdDecoder {
    type Result = DecodingResult;
    type Error = LdpcError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        self.decode(input)
    }

    fn check_count(&self) -> usize {
        self.check_count()
    }

    fn bit_count(&self) -> usize {
        self.bit_count()
    }
}

/// Implement Decoder trait for `UnionFindDecoder`
impl Decoder for UnionFindDecoder {
    type Result = DecodingResult;
    type Error = LdpcError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        self.decode(input, &[], 0)
    }

    fn check_count(&self) -> usize {
        self.check_count()
    }

    fn bit_count(&self) -> usize {
        self.bit_count()
    }
}

/// Implement Decoder trait for `FlipDecoder`
impl Decoder for FlipDecoder {
    type Result = DecodingResult;
    type Error = LdpcError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        self.decode(input)
    }

    fn check_count(&self) -> usize {
        self.check_count()
    }

    fn bit_count(&self) -> usize {
        self.bit_count()
    }
}

/// Implement Decoder trait for `BeliefFindDecoder`
impl Decoder for BeliefFindDecoder {
    type Result = DecodingResult;
    type Error = LdpcError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        self.decode(input)
    }

    fn check_count(&self) -> usize {
        self.check_count()
    }

    fn bit_count(&self) -> usize {
        self.bit_count()
    }
}

/// Implement Decoder trait for `SoftInfoBpDecoder` (special case)
impl Decoder for SoftInfoBpDecoder {
    type Result = DecodingResult;
    type Error = LdpcError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        // Convert u8 syndrome to f64 soft syndrome (0 -> 0.0, 1 -> 1.0)
        let soft_syndrome: Vec<f64> = input.iter().map(|&x| f64::from(x)).collect();
        // Use default parameters for cutoff and sigma
        self.decode(&soft_syndrome, 0.5, 1.0)
    }

    fn check_count(&self) -> usize {
        self.check_count()
    }

    fn bit_count(&self) -> usize {
        self.bit_count()
    }
}

/// Note: `UnionFindDecoder` has a complex decode signature that doesn't fit the basic pattern
/// Users would need to use it directly rather than through the Decoder trait
#[cfg(test)]
mod tests {

    // Note: These tests would require constructing LDPC decoders with proper parameters
    // which is complex, so we skip them for now. In practice, users would construct
    // the decoders with the appropriate parameters and then use the Decoder trait.
}
