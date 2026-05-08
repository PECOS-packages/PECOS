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

//! CSS-aware Union-Find decoder using the UIUF (Union-Intersection) algorithm.
//!
//! Exploits the CSS structure of surface codes by running UF independently on
//! X and Z syndrome graphs, then identifying likely Y errors via intersection
//! of the two cluster sets. Y errors are promoted to erasures, dramatically
//! improving accuracy (matching or exceeding MWPM).
//!
//! Reference: Tzu-Hao Lin and Ching-Yi Lai, "Union-Intersection Union-Find
//! Decoder," arXiv:2506.14745 (2025).
//!
//! This decoder takes TWO DEM strings (one for X-basis, one for Z-basis) and
//! decodes them jointly.

use crate::decoder::{UfDecoder, UfDecoderConfig};
use pecos_decoder_core::dem::{DemMatchingGraph, MatchingEdge};
use pecos_decoder_core::errors::DecoderError;

/// Compute the quantized spatial midpoint of an edge from detector coordinates.
///
/// For UIUF cross-graph matching, we use only the spatial coordinates
/// (first two dimensions) and ignore time (third dimension), since X and Z
/// stabilizers are measured at different times but share the same data qubits.
///
/// For same-time edges (timelike = measurement errors), the two endpoints
/// share spatial coords, so the midpoint is just that shared spatial position.
/// For space edges (data qubit errors), the midpoint is the data qubit's
/// spatial position.
///
/// Quantized to 0.001 resolution for use as a map key.
fn edge_spatial_midpoint(edge: &MatchingEdge, coords: &[Option<Vec<f64>>]) -> Option<(i64, i64)> {
    let c1 = coords.get(edge.node1 as usize)?.as_ref()?;

    if let Some(n2) = edge.node2 {
        let c2 = coords.get(n2 as usize)?.as_ref()?;
        // Spatial midpoint only (ignore time dimension).
        let x = ((c1.first().unwrap_or(&0.0) + c2.first().unwrap_or(&0.0)) * 500.0) as i64;
        let y = ((c1.get(1).unwrap_or(&0.0) + c2.get(1).unwrap_or(&0.0)) * 500.0) as i64;
        Some((x, y))
    } else {
        // Boundary edge.
        let x = (c1.first().unwrap_or(&0.0) * 1000.0) as i64;
        let y = (c1.get(1).unwrap_or(&0.0) * 1000.0) as i64;
        Some((x, y))
    }
}

/// Check if an edge is spatial (connects detectors at the same time)
/// vs timelike (connects detectors at different times).
/// Only spatial edges correspond to data qubit errors.
fn is_spatial_edge(edge: &MatchingEdge, coords: &[Option<Vec<f64>>]) -> bool {
    let Some(c1) = coords.get(edge.node1 as usize).and_then(|c| c.as_ref()) else {
        return true; // Assume spatial if no coords.
    };
    let Some(n2) = edge.node2 else {
        return true; // Boundary edges are spatial.
    };
    let Some(c2) = coords.get(n2 as usize).and_then(|c| c.as_ref()) else {
        return true;
    };
    let t1 = c1.get(2).unwrap_or(&0.0);
    let t2 = c2.get(2).unwrap_or(&0.0);
    (t1 - t2).abs() < 0.01 // Same time = spatial edge
}

/// Mapping of shared qubits between X and Z decoding graphs.
///
/// Each entry represents a data qubit that appears as an edge in both
/// the X and Z matching graphs. During UIUF intersection, if both
/// edges are covered by clusters, the qubit is marked as an erasure.
#[derive(Debug, Clone)]
pub struct QubitEdgeMapping {
    /// For each shared qubit: `(edge_idx in X graph, edge_idx in Z graph)`.
    pub pairs: Vec<(usize, usize)>,
}

/// CSS-aware UF decoder using UIUF intersection.
///
/// Wraps two `UfDecoder` instances (X and Z basis) and exploits the
/// overlap between their cluster sets to identify Y errors.
pub struct CssUfDecoder {
    /// UF decoder for X-basis syndromes (decodes Z errors).
    x_decoder: UfDecoder,
    /// UF decoder for Z-basis syndromes (decodes X errors).
    z_decoder: UfDecoder,
    /// Number of X detectors (split point for concatenated syndromes).
    x_num_detectors: usize,
    /// Qubit-to-edge mapping for intersection step.
    qubit_map: Option<QubitEdgeMapping>,
}

impl CssUfDecoder {
    /// Create from two DEM strings (X-basis and Z-basis).
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if either DEM is malformed.
    pub fn from_dems(
        x_dem: &str,
        z_dem: &str,
        config: UfDecoderConfig,
    ) -> Result<Self, DecoderError> {
        let x_graph = DemMatchingGraph::from_dem_str(x_dem)?;
        let z_graph = DemMatchingGraph::from_dem_str(z_dem)?;

        // Auto-detect qubit-edge mapping from detector coordinates.
        let qubit_map = Self::build_qubit_mapping(&x_graph, &z_graph);

        let x_num_detectors = x_graph.num_detectors;
        let x_decoder = UfDecoder::from_matching_graph(&x_graph, config);
        let z_decoder = UfDecoder::from_matching_graph(&z_graph, config);

        Ok(Self {
            x_decoder,
            z_decoder,
            x_num_detectors,
            qubit_map,
        })
    }

    /// Build qubit-edge mapping by matching spatial edge midpoints across graphs.
    ///
    /// Only spatial edges (same-time endpoints = data qubit errors) are matched.
    /// Timelike edges (measurement errors) are excluded since they don't
    /// correspond to shared data qubits.
    ///
    /// The mapping pairs edges in the X and Z graphs whose spatial midpoints
    /// coincide, identifying the shared data qubit.
    fn build_qubit_mapping(
        x_graph: &DemMatchingGraph,
        z_graph: &DemMatchingGraph,
    ) -> Option<QubitEdgeMapping> {
        use std::collections::BTreeMap;

        // Collect spatial edges from X graph, keyed by spatial midpoint.
        // Multiple X edges can share the same midpoint (different time slices).
        // We store all of them and match greedily.
        let mut x_midpoints: BTreeMap<(i64, i64), Vec<usize>> = BTreeMap::new();

        for (idx, edge) in x_graph.edges.iter().enumerate() {
            if !is_spatial_edge(edge, &x_graph.detector_coords) {
                continue;
            }
            if let Some(mid) = edge_spatial_midpoint(edge, &x_graph.detector_coords) {
                x_midpoints.entry(mid).or_default().push(idx);
            }
        }

        if x_midpoints.is_empty() {
            return None;
        }

        // Match Z spatial edges against X midpoints.
        let mut pairs = Vec::new();
        let mut used_x: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();

        for (z_idx, edge) in z_graph.edges.iter().enumerate() {
            if !is_spatial_edge(edge, &z_graph.detector_coords) {
                continue;
            }
            if let Some(mid) = edge_spatial_midpoint(edge, &z_graph.detector_coords)
                && let Some(x_candidates) = x_midpoints.get(&mid)
            {
                for &x_idx in x_candidates {
                    if !used_x.contains(&x_idx) {
                        pairs.push((x_idx, z_idx));
                        used_x.insert(x_idx);
                        break;
                    }
                }
            }
        }

        if pairs.is_empty() {
            None
        } else {
            Some(QubitEdgeMapping { pairs })
        }
    }

    /// Number of qubit pairs in the mapping (0 = no mapping, falls back to independent).
    #[must_use]
    pub fn num_qubit_pairs(&self) -> usize {
        self.qubit_map.as_ref().map_or(0, |m| m.pairs.len())
    }

    /// Set the qubit-to-edge mapping for UIUF intersection.
    ///
    /// Each pair `(x_edge_idx, z_edge_idx)` identifies a data qubit
    /// that appears as an edge in both the X and Z matching graphs.
    pub fn set_qubit_mapping(&mut self, mapping: QubitEdgeMapping) {
        self.qubit_map = Some(mapping);
    }

    /// Decode X and Z syndromes jointly using UIUF.
    ///
    /// If a qubit-to-edge mapping is set, uses the full UIUF algorithm
    /// (intersection to identify Y-error erasures). Otherwise falls back
    /// to independent UF decoding on each basis.
    ///
    /// Returns `(x_obs_mask, z_obs_mask)` -- observable predictions for each basis.
    ///
    /// # Errors
    ///
    /// Returns `DecoderError` if decoding fails.
    pub fn decode_css(
        &mut self,
        x_syndrome: &[u8],
        z_syndrome: &[u8],
    ) -> Result<(u64, u64), DecoderError> {
        if let Some(mapping) = &self.qubit_map {
            let pairs = mapping.pairs.clone();
            Ok(self.decode_uiuf(x_syndrome, z_syndrome, &pairs))
        } else {
            // Fallback: independent decoding.
            let x_obs = self.x_decoder.decode_syndrome(x_syndrome);
            let z_obs = self.z_decoder.decode_syndrome(z_syndrome);
            Ok((x_obs, z_obs))
        }
    }

    /// Count erasures that the intersection would produce (diagnostic).
    pub fn count_intersection_erasures(&mut self, x_syndrome: &[u8], z_syndrome: &[u8]) -> usize {
        if let Some(mapping) = &self.qubit_map {
            self.x_decoder.syndrome_validate(x_syndrome);
            self.z_decoder.syndrome_validate(z_syndrome);
            let mut count = 0;
            for &(x_edge, z_edge) in &mapping.pairs {
                let x_covered = self.x_decoder.edge_in_cluster(x_edge);
                let z_covered = self.z_decoder.edge_in_cluster(z_edge);
                if x_covered && z_covered {
                    count += 1;
                }
            }
            count
        } else {
            0
        }
    }

    /// Full UIUF algorithm.
    fn decode_uiuf(
        &mut self,
        x_syndrome: &[u8],
        z_syndrome: &[u8],
        qubit_pairs: &[(usize, usize)],
    ) -> (u64, u64) {
        // Phase 1: Syndrome validation (growth only) on each graph.
        self.x_decoder.syndrome_validate(x_syndrome);
        self.z_decoder.syndrome_validate(z_syndrome);

        // Phase 2: Intersection -- find edges covered in BOTH graphs.
        // These correspond to likely Y errors (which trigger both X and Z syndromes).
        let mut x_erasure_edges: Vec<usize> = Vec::new();
        let mut z_erasure_edges: Vec<usize> = Vec::new();

        for &(x_edge, z_edge) in qubit_pairs {
            let x_covered = self.x_decoder.edge_in_cluster(x_edge);
            let z_covered = self.z_decoder.edge_in_cluster(z_edge);
            if x_covered && z_covered {
                // Both graphs have clusters covering this qubit's edge.
                // Mark as erasure in both graphs for Phase 3.
                x_erasure_edges.push(x_edge);
                z_erasure_edges.push(z_edge);
            }
        }

        // Phase 3: Augmented UF decode with erasures.
        // X errors are decoded on Z graph (with Z erasures).
        // Z errors are decoded on X graph (with X erasures).
        let x_obs = self
            .x_decoder
            .decode_with_erasures(x_syndrome, &x_erasure_edges);
        let z_obs = self
            .z_decoder
            .decode_with_erasures(z_syndrome, &z_erasure_edges);

        (x_obs, z_obs)
    }
}

impl pecos_decoder_core::ObservableDecoder for CssUfDecoder {
    /// Decode a concatenated `[x_syndrome | z_syndrome]` via UIUF.
    ///
    /// The syndrome is split at `x_num_detectors` into X and Z parts.
    /// Returns the XOR of both observable masks.
    fn decode_to_observables(&mut self, syndrome: &[u8]) -> Result<u64, DecoderError> {
        let split = self.x_num_detectors;
        if syndrome.len() < split {
            return Err(DecoderError::DecodingFailed(format!(
                "CssUfDecoder: syndrome length {} < x_num_detectors {}",
                syndrome.len(),
                split
            )));
        }
        let x_syn = &syndrome[..split];
        let z_syn = &syndrome[split..];
        let (x_obs, z_obs) = self.decode_css(x_syn, z_syn)?;
        Ok(x_obs ^ z_obs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Minimal X-basis and Z-basis DEMs for a distance-3 repetition code.
    const X_DEM: &str = "\
error(0.01) D0 D1 L0
error(0.01) D0
error(0.01) D1
";

    const Z_DEM: &str = "\
error(0.01) D0 D1
error(0.01) D0 L0
error(0.01) D1
";

    #[test]
    fn test_css_construction() {
        let dec = CssUfDecoder::from_dems(X_DEM, Z_DEM, UfDecoderConfig::default());
        assert!(dec.is_ok());
    }

    #[test]
    fn test_css_no_errors() {
        let mut dec = CssUfDecoder::from_dems(X_DEM, Z_DEM, UfDecoderConfig::default()).unwrap();
        let (x_obs, z_obs) = dec.decode_css(&[0, 0], &[0, 0]).unwrap();
        assert_eq!(x_obs, 0);
        assert_eq!(z_obs, 0);
    }

    #[test]
    fn test_css_with_qubit_mapping() {
        let mut dec = CssUfDecoder::from_dems(X_DEM, Z_DEM, UfDecoderConfig::default()).unwrap();

        // Set up mapping: edge 0 in X_DEM corresponds to edge 0 in Z_DEM
        // (same data qubit connecting the two detectors).
        dec.set_qubit_mapping(QubitEdgeMapping {
            pairs: vec![(0, 0)],
        });

        // No errors: should still decode correctly.
        let (x_obs, z_obs) = dec.decode_css(&[0, 0], &[0, 0]).unwrap();
        assert_eq!(x_obs, 0);
        assert_eq!(z_obs, 0);

        // Y-error scenario: both X and Z syndromes have the same defects.
        // The intersection should identify the qubit as erasure.
        let (x_obs, z_obs) = dec.decode_css(&[1, 1], &[1, 1]).unwrap();
        // With erasure, both decoders should handle this correctly.
        // The exact observable depends on the DEM structure.
        let _ = (x_obs, z_obs); // Just verify no panic.
    }

    #[test]
    fn test_css_independent_decoding() {
        let mut dec = CssUfDecoder::from_dems(X_DEM, Z_DEM, UfDecoderConfig::default()).unwrap();

        // X syndrome has defects, Z syndrome clean.
        let (x_obs, z_obs) = dec.decode_css(&[1, 1], &[0, 0]).unwrap();
        assert_eq!(x_obs, 1); // D0-D1 edge carries L0 in X_DEM
        assert_eq!(z_obs, 0);

        // Z syndrome has defects, X syndrome clean.
        let (x_obs, z_obs) = dec.decode_css(&[0, 0], &[1, 1]).unwrap();
        assert_eq!(x_obs, 0);
        assert_eq!(z_obs, 0); // D0-D1 edge has no observable in Z_DEM
    }
}
