//! Relay BP and min-sum BP decoder implementations

use crate::config::{MinSumConfig, RelayConfig};
use crate::convert;
use crate::errors::{RelayBpError, Result};
use ndarray::{Array1, ArrayView1, ArrayView2};
use relay_bp::decoder::Decoder as RelayBpDecoderTrait;
use std::sync::Arc;

/// Decoding result from Relay BP / min-sum BP
#[derive(Debug, Clone)]
pub struct DecodingResult {
    /// The decoded error vector
    pub decoding: Array1<u8>,
    /// Whether the decoder converged
    pub converged: bool,
    /// Number of iterations used
    pub iterations: usize,
}

/// Relay BP ensemble decoder
///
/// Wraps `relay_bp::bp::relay::RelayDecoder<f64>`, which combines multiple
/// min-sum BP legs with disordered memory strengths for improved convergence
/// on qLDPC codes.
pub struct RelayBpDecoder {
    inner: relay_bp::bp::relay::RelayDecoder<f64>,
    num_checks: usize,
    num_bits: usize,
}

impl RelayBpDecoder {
    /// Create a new Relay BP decoder from a dense check matrix
    ///
    /// # Errors
    ///
    /// Returns [`RelayBpError::InvalidMatrix`] if the check matrix is invalid.
    pub fn new(
        check_matrix: &ArrayView2<u8>,
        min_sum_config: &MinSumConfig,
        relay_config: &RelayConfig,
    ) -> Result<Self> {
        let num_checks = check_matrix.nrows();
        let num_bits = check_matrix.ncols();

        let sparse_matrix = convert::check_matrix_to_relay(check_matrix)?;
        let ms_config = min_sum_config.to_min_sum_config();
        let relay_cfg = relay_config.to_relay_config();

        let inner = relay_bp::bp::relay::RelayDecoder::new(
            sparse_matrix,
            Arc::new(ms_config),
            Arc::new(relay_cfg),
        );

        Ok(Self {
            inner,
            num_checks,
            num_bits,
        })
    }

    /// Create a builder for configuring a new Relay BP decoder
    #[must_use]
    pub fn builder<'a>(check_matrix: &ArrayView2<'a, u8>) -> crate::builder::RelayBpBuilder<'a> {
        crate::builder::RelayBpBuilder::new(check_matrix)
    }

    /// Decode a syndrome vector
    ///
    /// # Errors
    ///
    /// Returns [`RelayBpError::InvalidSyndrome`] if the syndrome length doesn't
    /// match the number of checks.
    pub fn decode(&mut self, syndrome: &ArrayView1<u8>) -> Result<DecodingResult> {
        if syndrome.len() != self.num_checks {
            return Err(RelayBpError::InvalidSyndrome(format!(
                "Syndrome length {} doesn't match number of checks {}",
                syndrome.len(),
                self.num_checks
            )));
        }

        let relay_syndrome = convert::syndrome_to_relay(syndrome);
        let result = self.inner.decode_detailed(relay_syndrome.view());

        Ok(DecodingResult {
            decoding: convert::relay_array1_to_pecos(&result.decoding),
            converged: result.success,
            iterations: result.iterations,
        })
    }

    /// Get the number of checks (rows in parity check matrix)
    #[must_use]
    pub fn check_count(&self) -> usize {
        self.num_checks
    }

    /// Get the number of bits (columns in parity check matrix)
    #[must_use]
    pub fn bit_count(&self) -> usize {
        self.num_bits
    }
}

/// Min-sum BP decoder
///
/// Wraps `relay_bp::bp::min_sum::MinSumBPDecoder<f64>`, providing plain
/// min-sum belief propagation without the relay ensemble.
pub struct MinSumBpDecoder {
    inner: relay_bp::bp::min_sum::MinSumBPDecoder<f64>,
    num_checks: usize,
    num_bits: usize,
}

impl MinSumBpDecoder {
    /// Create a new min-sum BP decoder from a dense check matrix
    ///
    /// # Errors
    ///
    /// Returns [`RelayBpError::InvalidMatrix`] if the check matrix is invalid.
    pub fn new(check_matrix: &ArrayView2<u8>, config: &MinSumConfig) -> Result<Self> {
        let num_checks = check_matrix.nrows();
        let num_bits = check_matrix.ncols();

        let sparse_matrix = convert::check_matrix_to_relay(check_matrix)?;
        let ms_config = config.to_min_sum_config();

        let inner = relay_bp::bp::min_sum::MinSumBPDecoder::new(sparse_matrix, Arc::new(ms_config));

        Ok(Self {
            inner,
            num_checks,
            num_bits,
        })
    }

    /// Create a builder for configuring a new min-sum BP decoder
    #[must_use]
    pub fn builder<'a>(check_matrix: &ArrayView2<'a, u8>) -> crate::builder::MinSumBpBuilder<'a> {
        crate::builder::MinSumBpBuilder::new(check_matrix)
    }

    /// Decode a syndrome vector
    ///
    /// # Errors
    ///
    /// Returns [`RelayBpError::InvalidSyndrome`] if the syndrome length doesn't
    /// match the number of checks.
    pub fn decode(&mut self, syndrome: &ArrayView1<u8>) -> Result<DecodingResult> {
        if syndrome.len() != self.num_checks {
            return Err(RelayBpError::InvalidSyndrome(format!(
                "Syndrome length {} doesn't match number of checks {}",
                syndrome.len(),
                self.num_checks
            )));
        }

        let relay_syndrome = convert::syndrome_to_relay(syndrome);
        let result = self.inner.decode_detailed(relay_syndrome.view());

        Ok(DecodingResult {
            decoding: convert::relay_array1_to_pecos(&result.decoding),
            converged: result.success,
            iterations: result.iterations,
        })
    }

    /// Get the number of checks (rows in parity check matrix)
    #[must_use]
    pub fn check_count(&self) -> usize {
        self.num_checks
    }

    /// Get the number of bits (columns in parity check matrix)
    #[must_use]
    pub fn bit_count(&self) -> usize {
        self.num_bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    fn repetition_code_matrix() -> Array2<u8> {
        // H = [[1, 1, 0], [0, 1, 1]]
        Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap()
    }

    #[test]
    fn test_min_sum_decoder() {
        let h = repetition_code_matrix();
        let config = MinSumConfig::new(vec![0.1, 0.1, 0.1]);
        let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

        // Syndrome [1, 0] corresponds to error on first bit
        let syndrome = Array1::from_vec(vec![1u8, 0]);
        let result = decoder.decode(&syndrome.view()).unwrap();

        assert_eq!(result.decoding.len(), 3);
        assert!(result.converged);
    }

    #[test]
    fn test_relay_decoder() {
        let h = repetition_code_matrix();
        let ms_config = MinSumConfig::new(vec![0.1, 0.1, 0.1]);
        let relay_config = RelayConfig::default();
        let mut decoder = RelayBpDecoder::new(&h.view(), &ms_config, &relay_config).unwrap();

        let syndrome = Array1::from_vec(vec![1u8, 0]);
        let result = decoder.decode(&syndrome.view()).unwrap();

        assert_eq!(result.decoding.len(), 3);
        assert!(result.converged);
    }

    #[test]
    fn test_zero_syndrome() {
        let h = repetition_code_matrix();
        let config = MinSumConfig::new(vec![0.1, 0.1, 0.1]);
        let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

        let syndrome = Array1::from_vec(vec![0u8, 0]);
        let result = decoder.decode(&syndrome.view()).unwrap();

        assert_eq!(result.decoding.len(), 3);
        assert!(result.converged);
        // All zeros syndrome should produce all zeros decoding
        assert!(result.decoding.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_invalid_syndrome_length() {
        let h = repetition_code_matrix();
        let config = MinSumConfig::new(vec![0.1, 0.1, 0.1]);
        let mut decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

        let syndrome = Array1::from_vec(vec![1u8, 0, 1]);
        let result = decoder.decode(&syndrome.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_check_and_bit_count() {
        let h = repetition_code_matrix();
        let config = MinSumConfig::new(vec![0.1, 0.1, 0.1]);
        let decoder = MinSumBpDecoder::new(&h.view(), &config).unwrap();

        assert_eq!(decoder.check_count(), 2);
        assert_eq!(decoder.bit_count(), 3);
    }
}
