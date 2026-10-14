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

//! Complete edge corrections in the order supplied by the commit-window builder.

use crate::PyMatchingDecoder;
use pecos_decoder_core::window::CommitWindow;
use pecos_decoder_core::{DecoderError, EdgeDecoder};
use std::collections::BTreeMap;

/// Matching decoder with a construction-time map from endpoints to window edge indices.
pub struct PyMatchingEdgeDecoder {
    decoder: PyMatchingDecoder,
    edges: BTreeMap<(u32, Option<u32>), usize>,
}

impl PyMatchingEdgeDecoder {
    /// Replace the backend's error probabilities and derived matching weights.
    ///
    /// # Errors
    /// Returns an error for an invalid probability.
    pub fn set_all_error_probabilities(&mut self, probability: f64) -> Result<(), DecoderError> {
        self.decoder
            .set_all_error_probabilities(probability)
            .map_err(|error| DecoderError::InvalidConfiguration(error.to_string()))
    }

    /// Build from a window, retaining surviving error groups for correlated matching.
    ///
    /// # Errors
    /// Returns an error if the backend cannot construct the window graph.
    pub fn from_commit_window(
        window: &CommitWindow,
        correlated: bool,
    ) -> Result<Self, DecoderError> {
        let edges = window.edge_indices();
        let decoder = PyMatchingDecoder::from_dem_with_correlations(
            &window.model.to_dem_string(),
            correlated,
        )
        .map_err(|error| DecoderError::DecodingFailed(error.to_string()))?;
        Ok(Self { decoder, edges })
    }
}

impl EdgeDecoder for PyMatchingEdgeDecoder {
    fn decode_to_edges(&mut self, syndrome: &[u8]) -> Result<Vec<usize>, DecoderError> {
        self.decoder
            .decode_to_edges(syndrome)
            .map_err(|error| DecoderError::DecodingFailed(error.to_string()))?
            .into_iter()
            .map(|pair| {
                let node1 = u32::try_from(pair.detector1).map_err(|_| {
                    DecoderError::DecodingFailed(
                        "matching returned an invalid first endpoint".into(),
                    )
                })?;
                let node2 = pair.detector2.map(u32::try_from).transpose().map_err(|_| {
                    DecoderError::DecodingFailed(
                        "matching returned an invalid second endpoint".into(),
                    )
                })?;
                let key = match node2 {
                    Some(other) if other < node1 => (other, Some(node1)),
                    other => (node1, other),
                };
                self.edges.get(&key).copied().ok_or_else(|| {
                    DecoderError::DecodingFailed(format!(
                        "matching returned an edge absent from the commit window: {key:?}"
                    ))
                })
            })
            .collect()
    }
}
