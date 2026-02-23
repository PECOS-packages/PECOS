//! Implementation of core decoder traits for Relay BP decoders
//!
//! This module implements the standard traits from pecos-decoder-core
//! to ensure Relay BP decoders are compatible with the common decoder interface.

use crate::config::{MinSumConfig, RelayConfig};
use crate::decoder::{DecodingResult, MinSumBpDecoder, RelayBpDecoder};
use crate::errors::RelayBpError;
use ndarray::ArrayView1;
use pecos_decoder_core::{
    BatchDecoder, CheckMatrixConfig, CheckMatrixDecoder, Decoder, DecodingResultTrait,
    StandardDecodingResult,
};

// ============================================================================
// DecodingResultTrait
// ============================================================================

impl DecodingResultTrait for DecodingResult {
    fn is_successful(&self) -> bool {
        self.converged
    }

    fn cost(&self) -> Option<f64> {
        None
    }

    fn iterations(&self) -> Option<usize> {
        Some(self.iterations)
    }

    fn to_standard(&self) -> StandardDecodingResult {
        StandardDecodingResult {
            observable: self.decoding.to_vec(),
            weight: 0.0,
            converged: Some(self.converged),
            iterations: Some(self.iterations),
            confidence: None,
        }
    }
}

// ============================================================================
// Decoder trait
// ============================================================================

impl Decoder for RelayBpDecoder {
    type Result = DecodingResult;
    type Error = RelayBpError;

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

impl Decoder for MinSumBpDecoder {
    type Result = DecodingResult;
    type Error = RelayBpError;

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

// ============================================================================
// CheckMatrixDecoder trait
// ============================================================================

impl CheckMatrixDecoder for RelayBpDecoder {
    type CheckMatrixConfig = CheckMatrixConfig;

    fn from_dense_matrix_with_config(
        check_matrix: &ndarray::ArrayView2<u8>,
        config: Self::CheckMatrixConfig,
    ) -> Result<Self, pecos_decoder_core::DecoderError> {
        let ncols = check_matrix.ncols();

        // Use weights from config as error priors, or default to 0.1
        let error_priors = config.weights.unwrap_or_else(|| vec![0.1; ncols]);

        let ms_config = MinSumConfig::new(error_priors);
        let relay_config = RelayConfig::default();

        RelayBpDecoder::new(check_matrix, ms_config, relay_config)
            .map_err(pecos_decoder_core::DecoderError::from)
    }

    fn from_sparse_matrix_with_config(
        rows: Vec<usize>,
        cols: Vec<usize>,
        shape: (usize, usize),
        config: Self::CheckMatrixConfig,
    ) -> Result<Self, pecos_decoder_core::DecoderError> {
        let mut dense = ndarray::Array2::zeros(shape);
        for (&r, &c) in rows.iter().zip(cols.iter()) {
            dense[[r, c]] = 1;
        }
        Self::from_dense_matrix_with_config(&dense.view(), config)
    }
}

impl CheckMatrixDecoder for MinSumBpDecoder {
    type CheckMatrixConfig = CheckMatrixConfig;

    fn from_dense_matrix_with_config(
        check_matrix: &ndarray::ArrayView2<u8>,
        config: Self::CheckMatrixConfig,
    ) -> Result<Self, pecos_decoder_core::DecoderError> {
        let ncols = check_matrix.ncols();

        // Use weights from config as error priors, or default to 0.1
        let error_priors = config.weights.unwrap_or_else(|| vec![0.1; ncols]);

        let config = MinSumConfig::new(error_priors);

        MinSumBpDecoder::new(check_matrix, config).map_err(pecos_decoder_core::DecoderError::from)
    }

    fn from_sparse_matrix_with_config(
        rows: Vec<usize>,
        cols: Vec<usize>,
        shape: (usize, usize),
        config: Self::CheckMatrixConfig,
    ) -> Result<Self, pecos_decoder_core::DecoderError> {
        let mut dense = ndarray::Array2::zeros(shape);
        for (&r, &c) in rows.iter().zip(cols.iter()) {
            dense[[r, c]] = 1;
        }
        Self::from_dense_matrix_with_config(&dense.view(), config)
    }
}

// ============================================================================
// BatchDecoder trait
// ============================================================================

impl BatchDecoder for RelayBpDecoder {
    fn decode_batch(
        &mut self,
        inputs: &[ArrayView1<u8>],
    ) -> Result<Vec<Self::Result>, Self::Error> {
        inputs.iter().map(|s| self.decode(s)).collect()
    }
}

impl BatchDecoder for MinSumBpDecoder {
    fn decode_batch(
        &mut self,
        inputs: &[ArrayView1<u8>],
    ) -> Result<Vec<Self::Result>, Self::Error> {
        inputs.iter().map(|s| self.decode(s)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_decoder_trait_relay() {
        let h = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap();
        let ms_config = MinSumConfig::new(vec![0.1, 0.1, 0.1]);
        let relay_config = RelayConfig::default();
        let mut decoder = RelayBpDecoder::new(&h.view(), ms_config, relay_config).unwrap();

        let syndrome = Array1::from_vec(vec![1u8, 0]);
        let result = <RelayBpDecoder as Decoder>::decode(&mut decoder, &syndrome.view()).unwrap();

        assert_eq!(result.decoding.len(), 3);
        assert!(result.is_successful());
    }

    #[test]
    fn test_decoder_trait_min_sum() {
        let h = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap();
        let config = MinSumConfig::new(vec![0.1, 0.1, 0.1]);
        let mut decoder = MinSumBpDecoder::new(&h.view(), config).unwrap();

        let syndrome = Array1::from_vec(vec![0u8, 1]);
        let result = <MinSumBpDecoder as Decoder>::decode(&mut decoder, &syndrome.view()).unwrap();

        assert_eq!(result.decoding.len(), 3);
    }

    #[test]
    fn test_check_matrix_decoder_trait() {
        let config = CheckMatrixConfig {
            weights: Some(vec![0.1, 0.1, 0.1]),
            ..Default::default()
        };

        let h = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap();
        let decoder = MinSumBpDecoder::from_dense_matrix_with_config(&h.view(), config).unwrap();

        assert_eq!(decoder.check_count(), 2);
        assert_eq!(decoder.bit_count(), 3);
    }

    #[test]
    fn test_batch_decoder_trait() {
        let h = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap();
        let config = MinSumConfig::new(vec![0.1, 0.1, 0.1]);
        let mut decoder = MinSumBpDecoder::new(&h.view(), config).unwrap();

        let s1 = Array1::from_vec(vec![1u8, 0]);
        let s2 = Array1::from_vec(vec![0u8, 1]);
        let syndromes = vec![s1.view(), s2.view()];

        let results = decoder.decode_batch(&syndromes).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_decoding_result_trait() {
        let result = DecodingResult {
            decoding: Array1::from_vec(vec![1, 0, 0]),
            converged: true,
            iterations: 5,
        };

        assert!(result.is_successful());
        assert_eq!(result.iterations(), Some(5));
        let std_result = result.to_standard();
        assert_eq!(std_result.converged, Some(true));
        assert_eq!(std_result.iterations, Some(5));
    }
}
