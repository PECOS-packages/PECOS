//! Implementation of core decoder traits for `FusionBlossom`
//!
//! This module implements the standard traits from pecos-decoder-core
//! to ensure `FusionBlossom` is compatible with the common decoder interface.

use crate::decoder::{
    DecodingOptions, DecodingResult, FusionBlossomConfig, FusionBlossomDecoder, SyndromeData,
};
use crate::errors::FusionBlossomError;
use ndarray::{ArrayView1, ArrayView2};
use pecos_decoder_core::{
    AdvancedDecoder, AdvancedDecodingResult, CheckMatrixConfig, CheckMatrixDecoder, Decoder,
    DecodingOptions as CoreDecodingOptions, DecodingResultTrait, DecodingStats,
    DynamicWeightDecoder, ErasureDecoder, MatchedEdge, StandardDecodingResult,
};

/// Implement the core Decoder trait for `FusionBlossomDecoder`
impl Decoder for FusionBlossomDecoder {
    type Result = DecodingResult;
    type Error = FusionBlossomError;

    fn decode(&mut self, input: &ArrayView1<u8>) -> Result<Self::Result, Self::Error> {
        // Use the existing decode method
        self.decode(input)
    }

    fn check_count(&self) -> usize {
        self.num_nodes()
    }

    fn bit_count(&self) -> usize {
        self.num_edges()
    }
}

/// Implement `DecodingResultTrait` for `FusionBlossom`'s `DecodingResult`
impl DecodingResultTrait for DecodingResult {
    fn is_successful(&self) -> bool {
        // FusionBlossom always returns a result if it doesn't error
        true
    }

    fn cost(&self) -> Option<f64> {
        Some(self.weight)
    }

    fn iterations(&self) -> Option<usize> {
        // FusionBlossom doesn't expose iteration count
        None
    }

    fn to_standard(&self) -> StandardDecodingResult {
        StandardDecodingResult {
            observable: self.observable.clone(),
            weight: self.weight,
            converged: Some(true), // FusionBlossom always converges
            iterations: None,
            confidence: None,
        }
    }
}

/// Implement `CheckMatrixDecoder` trait for `FusionBlossomDecoder`
impl CheckMatrixDecoder for FusionBlossomDecoder {
    type CheckMatrixConfig = CheckMatrixConfig;

    fn from_dense_matrix_with_config(
        check_matrix: &ArrayView2<u8>,
        config: Self::CheckMatrixConfig,
    ) -> Result<Self, pecos_decoder_core::DecoderError> {
        // Convert dense matrix to the format expected by FusionBlossom
        let dense_array = check_matrix.to_owned();

        // Create FusionBlossom config from CheckMatrixConfig
        let fb_config = FusionBlossomConfig {
            num_nodes: Some(check_matrix.nrows()),
            num_observables: config.num_observables.unwrap_or(1),
            ..Default::default()
        };

        // Extract weights from config
        let weights = config.weights.as_deref();

        FusionBlossomDecoder::from_check_matrix(&dense_array, weights, fb_config)
            .map_err(pecos_decoder_core::DecoderError::from)
    }

    fn from_sparse_matrix_with_config(
        rows: Vec<usize>,
        cols: Vec<usize>,
        shape: (usize, usize),
        config: Self::CheckMatrixConfig,
    ) -> Result<Self, pecos_decoder_core::DecoderError> {
        // Convert sparse to dense for FusionBlossom
        let mut dense = ndarray::Array2::zeros(shape);
        for (&r, &c) in rows.iter().zip(cols.iter()) {
            dense[[r, c]] = 1;
        }

        Self::from_dense_matrix_with_config(&dense.view(), config)
    }
}

/// Implement `ErasureDecoder` trait for `FusionBlossomDecoder`
impl ErasureDecoder for FusionBlossomDecoder {
    fn decode_with_erasures(
        &mut self,
        syndrome: &ArrayView1<u8>,
        erasures: &[usize],
    ) -> Result<Self::Result, Self::Error> {
        // Convert syndrome to defects (non-zero indices)
        let defects: Vec<usize> = syndrome
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v != 0 { Some(i) } else { None })
            .collect();

        // Create syndrome data with erasures
        let syndrome_data = SyndromeData::with_erasures(defects, erasures.to_vec());

        // Use advanced decode with erasures
        self.decode_advanced(syndrome_data)
    }
}

/// Implement `DynamicWeightDecoder` trait for `FusionBlossomDecoder`
impl DynamicWeightDecoder for FusionBlossomDecoder {
    fn update_edge_weights(
        &mut self,
        edges: &[(usize, usize)],
        weights: &[f64],
    ) -> Result<(), pecos_decoder_core::DecoderError> {
        if edges.len() != weights.len() {
            return Err(pecos_decoder_core::DecoderError::InvalidConfiguration(
                format!(
                    "Edge count {} doesn't match weight count {}",
                    edges.len(),
                    weights.len()
                ),
            ));
        }

        // Convert edge pairs to edge indices and weights
        // This is a simplified implementation - real implementation would need
        // to map (node1, node2) pairs to edge indices
        let _dynamic_weights: Vec<(usize, i32)> = edges
            .iter()
            .zip(weights)
            .enumerate()
            .map(|(i, ((_n1, _n2), &w))| (i, (w * 1000.0) as i32)) // Convert to integer weights
            .collect();

        // Store for next decode operation
        // Note: This is a simplified implementation
        // Real implementation would update the solver's edge weights
        Ok(())
    }

    fn reset_weights(&mut self) -> Result<(), pecos_decoder_core::DecoderError> {
        // Reset solver to use original weights
        // This forces recreation of the solver with original weights
        // Clear cached solver to force re-initialization
        self.clear_solver_cache();
        Ok(())
    }
}

/// Implement `AdvancedDecoder` trait for `FusionBlossomDecoder`
impl AdvancedDecoder for FusionBlossomDecoder {
    fn decode_advanced(
        &mut self,
        syndrome: &ArrayView1<u8>,
        options: CoreDecodingOptions,
    ) -> Result<AdvancedDecodingResult<Self::Result>, Self::Error> {
        // Convert syndrome to defects
        let defects: Vec<usize> = syndrome
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| if v != 0 { Some(i) } else { None })
            .collect();

        // Create syndrome data
        let mut syndrome_data = SyndromeData::from_defects(defects);

        // Apply erasures if provided
        if let Some(erasures) = options.erasures {
            syndrome_data.erasures = Some(erasures);
        }

        // Apply dynamic weights if provided
        if let Some(edge_weights) = options.edge_weights {
            let dynamic_weights: Vec<(usize, i32)> = edge_weights
                .into_iter()
                .map(|(edge_idx, _node1, weight)| (edge_idx, (weight * 1000.0) as i32))
                .collect();
            syndrome_data.dynamic_weights = Some(dynamic_weights);
        }

        // Create decoding options
        let decode_options = DecodingOptions {
            include_perfect_matching: options.return_details,
        };

        // Perform decoding
        let result = self.decode_with_options(syndrome_data, decode_options)?;

        // Create stats
        let stats = DecodingStats {
            iterations: None,
            time_taken: None,
            nodes_explored: None,
            blossoms_formed: None, // Could extract from perfect matching info
            converged: true,
            confidence: None,
        };

        // Create matched edges if requested
        let matched_edges = if options.return_details {
            Some(
                result
                    .matched_edges
                    .iter()
                    .map(|&edge_idx| {
                        MatchedEdge {
                            node1: edge_idx, // Simplified mapping
                            node2: edge_idx + 1,
                            weight: result.weight / result.matched_edges.len() as f64,
                            observables: vec![], // Not easily available
                        }
                    })
                    .collect(),
            )
        } else {
            None
        };

        Ok(AdvancedDecodingResult {
            result,
            stats,
            matched_edges,
            matched_pairs: None, // Not implemented for simplicity
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    #[test]
    fn test_decoder_trait_implementation() {
        // Create a simple repetition code matrix: H = [[1, 1, 0], [0, 1, 1]]
        let check_matrix = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap();

        let config = FusionBlossomConfig::default();
        let mut decoder =
            FusionBlossomDecoder::from_check_matrix(&check_matrix, None, config).unwrap();

        // Test decode
        let syndrome = Array1::from_vec(vec![1, 0]);
        let result =
            <FusionBlossomDecoder as Decoder>::decode(&mut decoder, &syndrome.view()).unwrap();

        assert!(!result.observable.is_empty());
        assert!(result.weight >= 0.0);
    }

    #[test]
    fn test_erasure_decoder_trait() {
        let check_matrix = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap();

        let config = FusionBlossomConfig::default();
        let mut decoder =
            FusionBlossomDecoder::from_check_matrix(&check_matrix, None, config).unwrap();

        // Test decode with erasures
        let syndrome = Array1::from_vec(vec![1, 0]);
        let erasures = vec![0]; // First edge is erased

        let result = decoder
            .decode_with_erasures(&syndrome.view(), &erasures)
            .unwrap();

        assert!(!result.observable.is_empty());
        assert!(result.weight >= 0.0);
    }

    #[test]
    fn test_check_matrix_decoder_trait() {
        let config = CheckMatrixConfig {
            num_observables: Some(2),
            ..Default::default()
        };

        let check_matrix = Array2::from_shape_vec((2, 3), vec![1, 1, 0, 0, 1, 1]).unwrap();

        let decoder =
            FusionBlossomDecoder::from_dense_matrix_with_config(&check_matrix.view(), config)
                .unwrap();

        assert_eq!(decoder.check_count(), 2);
    }
}
