//! Implementation of core decoder traits for `PyMatching`
//!
//! This module implements the standard traits from pecos-decoder-core
//! to ensure `PyMatching` is compatible with the common decoder interface.

use crate::decoder::{CheckMatrix, CheckMatrixConfig, DecodingResult, PyMatchingDecoder};
use crate::errors::PyMatchingError;
use ndarray::{ArrayView1, ArrayView2};
use pecos_decoder_core::{
    BatchDecoder, CheckMatrixDecoder, Decoder, DecodingStats, DemDecoder, DetailedDecoder,
    MatchedEdge, MatchedPair as CoreMatchedPair,
};

/// Implement the core Decoder trait for `PyMatchingDecoder`
impl Decoder for PyMatchingDecoder {
    type Result = DecodingResult;
    type Error = PyMatchingError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        // Convert ArrayView to slice and call existing decode method
        self.decode(input.as_slice().ok_or_else(|| {
            PyMatchingError::Configuration("Input must be contiguous".to_string())
        })?)
    }

    fn check_count(&self) -> usize {
        self.num_nodes()
    }

    fn bit_count(&self) -> usize {
        // For PyMatching, this is the number of error mechanisms
        // which is typically the number of edges in the original graph
        self.num_edges()
    }
}

// DecodingResultTrait is already implemented in decoder.rs

/// Implement `CheckMatrixDecoder` trait for `PyMatchingDecoder`
impl CheckMatrixDecoder for PyMatchingDecoder {
    type CheckMatrixConfig = CheckMatrixConfig;

    fn from_dense_matrix_with_config(
        check_matrix: &ArrayView2<u8>,
        mut config: Self::CheckMatrixConfig,
    ) -> Result<Self, pecos_decoder_core::DecoderError> {
        // Convert dense matrix to CheckMatrix format
        let rows = check_matrix.nrows();
        let _cols = check_matrix.ncols();

        let dense_vec: Vec<Vec<u8>> = (0..rows).map(|r| check_matrix.row(r).to_vec()).collect();

        let mut matrix = CheckMatrix::from_dense_vec(&dense_vec)
            .map_err(pecos_decoder_core::DecoderError::from)?;

        // Apply configuration if provided
        if let Some(weights) = config.weights.take() {
            matrix = matrix
                .with_weights(weights)
                .map_err(pecos_decoder_core::DecoderError::from)?;
        }

        PyMatchingDecoder::from_check_matrix_with_config(&matrix, config)
            .map_err(pecos_decoder_core::DecoderError::from)
    }

    fn from_sparse_matrix_with_config(
        rows: Vec<usize>,
        cols: Vec<usize>,
        shape: (usize, usize),
        mut config: Self::CheckMatrixConfig,
    ) -> Result<Self, pecos_decoder_core::DecoderError> {
        // Create CheckMatrix from sparse format
        let mut matrix = CheckMatrix::new(shape.0, shape.1, rows, cols);

        // Apply configuration if provided
        if let Some(weights) = config.weights.take() {
            matrix = matrix
                .with_weights(weights)
                .map_err(pecos_decoder_core::DecoderError::from)?;
        }

        PyMatchingDecoder::from_check_matrix_with_config(&matrix, config)
            .map_err(pecos_decoder_core::DecoderError::from)
    }
}

/// Implement `DemDecoder` trait for `PyMatchingDecoder`
impl DemDecoder for PyMatchingDecoder {
    type DemConfig = (); // PyMatching doesn't have DEM-specific config

    fn from_dem_with_config(
        dem: &str,
        _config: Self::DemConfig,
    ) -> Result<Self, pecos_decoder_core::DecoderError> {
        PyMatchingDecoder::from_dem(dem).map_err(pecos_decoder_core::DecoderError::from)
    }

    fn detector_count(&self) -> usize {
        self.num_detectors()
    }

    fn observable_count(&self) -> usize {
        self.num_observables()
    }
}

/// Implement `BatchDecoder` trait for `PyMatchingDecoder`
impl BatchDecoder for PyMatchingDecoder {
    fn decode_batch(
        &mut self,
        inputs: &[ArrayView1<u8>],
    ) -> Result<Vec<Self::Result>, Self::Error> {
        // PyMatching doesn't have a simple batch interface, so we decode one by one
        inputs
            .iter()
            .map(|input| <Self as Decoder>::decode(self, input))
            .collect()
    }
}

/// Implement `DetailedDecoder` trait for `PyMatchingDecoder`
impl DetailedDecoder for PyMatchingDecoder {
    fn decode_to_edges(
        &mut self,
        syndrome: &ArrayView1<u8>,
    ) -> Result<Vec<MatchedEdge>, Self::Error> {
        // First decode to get the result with weight
        let _decode_result = <Self as Decoder>::decode(self, syndrome)?;

        // Then get the matched pairs
        let pairs = self.decode_to_matched_pairs(syndrome.as_slice().ok_or_else(|| {
            PyMatchingError::Configuration("Input must be contiguous".to_string())
        })?)?;

        // Convert MatchedPair to MatchedEdge
        // Note: PyMatching's MatchedPair doesn't include per-edge weights or observables
        Ok(pairs
            .into_iter()
            .map(|pair| {
                MatchedEdge {
                    node1: pair.detector1 as usize,
                    node2: pair
                        .detector2
                        .map_or(crate::decoder::BOUNDARY_NODE_MARKER, |d| d as usize),
                    weight: 0.0,         // Individual edge weights not available
                    observables: vec![], // Observable info not available per edge
                }
            })
            .collect())
    }

    fn decode_to_pairs(
        &mut self,
        syndrome: &ArrayView1<u8>,
    ) -> Result<Vec<CoreMatchedPair>, Self::Error> {
        let pairs = self.decode_to_matched_pairs(syndrome.as_slice().ok_or_else(|| {
            PyMatchingError::Configuration("Input must be contiguous".to_string())
        })?)?;

        // Convert to core MatchedPair type
        Ok(pairs
            .into_iter()
            .map(|pair| {
                CoreMatchedPair {
                    detector1: pair.detector1 as usize,
                    detector2: pair.detector2.map(|d| d as usize),
                    weight: 0.0, // Individual pair weights not available
                }
            })
            .collect())
    }

    fn get_stats(&self) -> DecodingStats {
        // PyMatching doesn't expose detailed stats
        DecodingStats {
            iterations: None,
            time_taken: None,
            nodes_explored: None,
            blossoms_formed: None,
            converged: true,
            confidence: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_decoder_trait_implementation() {
        // Create a simple repetition code
        let check_matrix = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap();

        let mut decoder = PyMatchingDecoder::from_dense_matrix(&check_matrix.view()).unwrap();

        // Test decode
        let syndrome = Array1::from_vec(vec![1, 0]);
        let result =
            <PyMatchingDecoder as Decoder>::decode(&mut decoder, &syndrome.view()).unwrap();

        // PyMatching returns one bit per observable
        assert!(!result.observable.is_empty());
        assert!(result.weight >= 0.0);
    }

    #[test]
    fn test_check_matrix_decoder_trait() {
        let config = CheckMatrixConfig {
            weights: Some(vec![1.0, 2.0, 1.0]),
            ..Default::default()
        };

        let check_matrix = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap();

        let decoder =
            PyMatchingDecoder::from_dense_matrix_with_config(&check_matrix.view(), config).unwrap();

        assert_eq!(decoder.check_count(), 2);
    }
}
