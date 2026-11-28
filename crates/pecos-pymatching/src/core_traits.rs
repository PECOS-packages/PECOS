//! Implementation of core decoder traits for `PyMatching`
//!
//! This module implements the standard traits from pecos-decoder-core
//! to ensure `PyMatching` is compatible with the common decoder interface.

use crate::decoder::{CheckMatrix, CheckMatrixConfig, DecodingResult, PyMatchingDecoder};
use crate::errors::PyMatchingError;
use ndarray::{ArrayView1, ArrayView2};
use pecos_decoder_core::{
    BatchDecoder, CheckMatrixDecoder, Decoder, DecoderError, DecodingStats, DemDecoder,
    DetailedDecoder, MatchedEdge, MatchedPair as CoreMatchedPair, ObservableDecoder,
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
        let pairs = PyMatchingDecoder::decode_to_edges(
            self,
            syndrome.as_slice().ok_or_else(|| {
                PyMatchingError::Configuration("Input must be contiguous".to_string())
            })?,
        )?;

        pairs
            .into_iter()
            .map(|pair| {
                let node1 = pair.detector1 as usize;
                let data = match pair.detector2 {
                    Some(node2) => self.get_edge_data(node1, node2 as usize)?,
                    None => self.get_boundary_edge_data(node1)?,
                };
                Ok(MatchedEdge {
                    node1,
                    node2: data.node2.unwrap_or(crate::decoder::BOUNDARY_NODE_MARKER),
                    weight: data.weight,
                    observables: data.observables,
                })
            })
            .collect()
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

/// Implement `ObservableDecoder` for `PyMatchingDecoder`.
///
/// Converts the observable vector to a bitmask for the sample+decode loop.
impl ObservableDecoder for PyMatchingDecoder {
    fn decode_obs(
        &mut self,
        syndrome: &[u8],
    ) -> Result<pecos_decoder_core::obs_mask::ObsMask, DecoderError> {
        // `decode` rejects a length outside upstream PyMatching's dense range.
        let result = self.decode(syndrome).map_err(DecoderError::from)?;
        let mut mask = pecos_decoder_core::obs_mask::ObsMask::new();
        for (i, &v) in result.observable.iter().enumerate() {
            if v != 0 {
                mask.set(i);
            }
        }
        Ok(mask)
    }

    fn decode_batch_to_observables(
        &mut self,
        shots: &[u8],
        num_shots: usize,
        num_detectors: usize,
    ) -> Result<Vec<pecos_decoder_core::obs_mask::ObsMask>, DecoderError> {
        use crate::decoder::BatchConfig;
        // `decode_batch_with_config` validates the width, even for zero shots.
        let config = BatchConfig {
            bit_packed_input: false,
            bit_packed_output: true,
            return_weights: false,
        };
        let result = self
            .decode_batch_with_config(shots, num_shots, num_detectors, config)
            .map_err(DecoderError::from)?;

        // Convert each bit-packed prediction into little-endian words. The
        // bridge already emits every observable byte, including above bit 63.
        let mut masks = Vec::with_capacity(num_shots);
        for pred in &result.predictions {
            let mut words = vec![0u64; pred.len().div_ceil(8)];
            for (byte_index, &byte) in pred.iter().enumerate() {
                words[byte_index / 8] |= u64::from(byte) << ((byte_index % 8) * 8);
            }
            masks.push(pecos_decoder_core::obs_mask::ObsMask::from_words(&words));
        }
        Ok(masks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2};

    fn assert_wrong_length(
        result: &Result<pecos_decoder_core::obs_mask::ObsMask, DecoderError>,
        expected: usize,
        actual: usize,
    ) {
        assert!(
            matches!(
                result,
                Err(DecoderError::InvalidDimensions { expected: e, actual: a })
                    if *e == expected && *a == actual
            ),
            "expected InvalidDimensions {{ expected: {expected}, actual: {actual} }}, got {result:?}"
        );
    }

    #[test]
    fn observable_decode_rejects_short_and_long_syndromes() {
        let mut decoder =
            crate::PyMatchingDecoder::from_dem("error(0.1) D0 L0\nerror(0.1) D0 D1\ndetector D2")
                .unwrap();
        for length in [0, 2, 4] {
            assert_wrong_length(
                &pecos_decoder_core::ObservableDecoder::decode_obs(&mut decoder, &vec![0; length]),
                3,
                length,
            );
        }
        assert!(pecos_decoder_core::ObservableDecoder::decode_obs(&mut decoder, &[0; 3]).is_ok());
    }

    #[test]
    fn set_boundary_refreshes_detector_count() {
        let mut decoder = crate::PyMatchingDecoder::from_dem(
            "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1 D2",
        )
        .unwrap();
        assert_eq!(decoder.num_detectors(), 3);
        decoder.set_boundary(&[2]).unwrap();
        assert_eq!(decoder.num_detectors(), 2);
        // As upstream, a dense syndrome may also carry the boundary node's entry.
        for length in [2, 3] {
            let mut syndrome = vec![0; length];
            syndrome[0] = 1;
            assert!(
                pecos_decoder_core::ObservableDecoder::decode_obs(&mut decoder, &syndrome).is_ok()
            );
        }
        for length in [1, 4] {
            let result =
                pecos_decoder_core::ObservableDecoder::decode_obs(&mut decoder, &vec![0; length]);
            assert!(
                matches!(
                    &result,
                    Err(DecoderError::InvalidSyndrome(message))
                        if message == &format!("dense syndrome has {length} entries, but this graph accepts 2 (one per detector) to 3 (one per node, boundary nodes included)")
                ),
                "{result:?}"
            );
        }
    }

    #[test]
    fn non_trailing_boundary_leaves_every_detector_addressable() {
        // Node 0 is the boundary, so the detectors are nodes 1 and 2 and dense
        // entry 2 is the only way to address the top detector.
        let mut decoder =
            crate::PyMatchingDecoder::from_dem("error(0.1) D0 D1\nerror(0.1) D1 D2 L0").unwrap();
        decoder.set_boundary(&[0]).unwrap();
        assert_eq!(decoder.num_detectors(), 2);
        assert_eq!(decoder.num_nodes(), 3);
        // Detector 2 matches to the boundary through D1 D2, which flips L0.
        assert_eq!(decoder.decode(&[0, 0, 1]).unwrap().observable, vec![1]);
        let mask =
            pecos_decoder_core::ObservableDecoder::decode_obs(&mut decoder, &[0, 0, 1]).unwrap();
        assert!(mask.get(0));
        let batch = pecos_decoder_core::ObservableDecoder::decode_batch_to_observables(
            &mut decoder,
            &[0, 0, 1],
            1,
            3,
        )
        .unwrap();
        assert!(batch[0].get(0));
        // Detector 1 alone matches straight to the boundary.
        assert_eq!(decoder.decode(&[0, 1]).unwrap().observable, vec![0]);
    }

    #[test]
    fn upstream_dem_rejection_is_an_invalid_graph() {
        // PyMatching throws `std::invalid_argument` for a decomposed error with an
        // undetectable component; the bridge reports it as an input rejection.
        let dem = "error(0.1) D0 ^ L0\ndetector D0";
        let error = crate::PyMatchingDecoder::from_dem_with_correlations(dem, true)
            .err()
            .expect("correlated PyMatching rejects an undetectable component");
        assert!(
            matches!(&error, crate::PyMatchingError::InvalidGraph(message)
                if message.starts_with("Failed to parse DEM: ")
                    && message.contains("undetectable component")),
            "{error:?}"
        );
        let error = DecoderError::from(error);
        assert!(
            matches!(&error, DecoderError::InvalidGraph(message)
                if message.contains("undetectable component")),
            "{error:?}"
        );
    }

    #[test]
    fn upstream_weight_rejection_is_an_invalid_graph_when_the_matcher_is_built() {
        // `error(1)` builds, but its infinite weight is rejected when the matcher
        // is first built for decoding.
        let mut decoder =
            crate::PyMatchingDecoder::from_dem("error(1) D0").expect("the graph builds");
        let error = decoder
            .decode(&[1])
            .expect_err("upstream rejects the edge weight");
        assert!(
            matches!(&error, crate::PyMatchingError::InvalidGraph(message)
                if message.starts_with("Failed to build the matcher: ")
                    && message.contains("maximum absolute edge weight")),
            "{error:?}"
        );
    }

    #[test]
    fn error_probability_override_replaces_an_out_of_range_dem_weight() {
        // The override rewrites every edge weight before the matcher is built, so
        // the DEM's own infinite weight is never checked. (Implied correlation
        // weights are kept from the DEM; this graph is uncorrelated.)
        let mut decoder =
            crate::PyMatchingDecoder::from_dem_with_error_probability("error(1) D0", 0.1)
                .expect("the graph builds");
        decoder
            .decode(&[1])
            .expect("the overridden weight is in range");
    }

    /// A graph built edge by edge whose weight upstream only rejects when it
    /// builds the matcher, with `num_observables` observables.
    fn overweight_graph(num_observables: usize) -> crate::PyMatchingDecoder {
        let mut decoder = crate::PyMatchingDecoder::new(crate::PyMatchingConfig {
            num_neighbours: None,
            num_nodes: Some(2),
            num_observables,
        })
        .expect("the graph is created");
        decoder
            .add_edge(0, 1, &[0], Some(1e12), None, None)
            .expect("add_edge stores any finite weight");
        decoder
    }

    fn assert_matcher_build_rejected<T: std::fmt::Debug>(result: &crate::errors::Result<T>) {
        assert!(
            matches!(result, Err(crate::PyMatchingError::InvalidGraph(message))
                if message.starts_with("Failed to build the matcher: ")
                    && message.contains("maximum absolute edge weight")),
            "{result:?}"
        );
    }

    #[test]
    fn matcher_build_rejection_at_decode_is_an_invalid_graph_on_every_entry_point() {
        let syndrome = [1, 1];
        assert_matcher_build_rejected(&overweight_graph(1).decode(&syndrome));
        assert_matcher_build_rejected(&overweight_graph(65).decode(&syndrome));
        assert_matcher_build_rejected(&overweight_graph(1).decode_to_matched_pairs(&syndrome));
        assert_matcher_build_rejected(&overweight_graph(1).decode_to_edges(&syndrome));
        assert_matcher_build_rejected(&overweight_graph(1).decode_batch_with_config(
            &syndrome,
            1,
            2,
            crate::BatchConfig::default(),
        ));
    }

    #[test]
    fn graph_creation_and_boundary_failures_are_invalid_graphs() {
        // 2**62 nodes fail the allocation size check at once (`std::length_error`).
        for num_observables in [1, 65] {
            let error = crate::PyMatchingDecoder::new(crate::PyMatchingConfig {
                num_neighbours: None,
                num_nodes: Some(1 << 62),
                num_observables,
            })
            .err()
            .expect("the graph is too large to create");
            assert!(
                matches!(&error, crate::PyMatchingError::InvalidGraph(message)
                    if message.starts_with("Failed to create the graph: ")),
                "{error:?}"
            );
        }
        let mut decoder =
            crate::PyMatchingDecoder::from_dem("error(0.1) D0 D1").expect("the graph builds");
        for node in [usize::MAX, 1 << 62] {
            let error = decoder
                .set_boundary(&[node])
                .expect_err("the boundary node is out of range");
            assert!(
                matches!(error, crate::PyMatchingError::InvalidGraph(_)),
                "{error:?}"
            );
        }
        assert_eq!(decoder.num_nodes(), 2);
        assert!(!decoder.is_boundary_node(2));
        assert!(!decoder.is_boundary_node(1 << 40));
        assert!(decoder.is_boundary_node(usize::MAX));
    }

    #[test]
    fn decode_phase_failure_stays_an_ffi_error() {
        // An odd-parity syndrome with no boundary has no perfect matching.
        let mut decoder =
            crate::PyMatchingDecoder::from_dem("error(0.1) D0 D1").expect("the graph builds");
        let error = decoder
            .decode(&[1, 0])
            .expect_err("no perfect matching exists");
        assert!(matches!(error, crate::PyMatchingError::Ffi(_)), "{error:?}");
    }

    #[test]
    fn out_of_range_weight_normalising_constant_is_an_invalid_graph() {
        // Upstream throws `std::invalid_argument` for an edge weight above its
        // maximum; it must reach Rust as an error, not end the process. DEM
        // construction now rejects such weights, so build the graph edge by edge.
        let decoder = overweight_graph(1);
        let error = decoder
            .get_edge_weight_normalising_constant(1000)
            .expect_err("upstream rejects the edge weight");
        assert!(
            matches!(&error, crate::PyMatchingError::InvalidGraph(message)
                if message.contains("maximum absolute edge weight")),
            "{error:?}"
        );
    }

    fn decoder_with_nodes(num_nodes: usize) -> crate::PyMatchingDecoder {
        crate::PyMatchingDecoder::new(crate::PyMatchingConfig {
            num_neighbours: None,
            num_nodes: Some(num_nodes),
            num_observables: 1,
        })
        .expect("the graph is created")
    }

    fn assert_invalid_graph<T>(result: &crate::errors::Result<T>, needle: &str) {
        match result {
            Err(crate::PyMatchingError::InvalidGraph(message)) if message.contains(needle) => {}
            Err(error) => panic!("expected an invalid graph naming {needle:?}, got {error:?}"),
            Ok(_) => panic!("expected an invalid graph naming {needle:?}, got Ok"),
        }
    }

    #[test]
    fn boundary_node_marker_is_rejected_as_an_edge_node_before_the_edge_is_added() {
        use crate::decoder::BOUNDARY_NODE_MARKER;
        // Upstream sizes its node list as `node + 1`, which wraps for this node,
        // and then indexes past the end.
        let mut decoder = decoder_with_nodes(4);
        let marker = "BOUNDARY_NODE_MARKER (usize::MAX) is not a node";
        assert_invalid_graph(
            &decoder.add_edge(BOUNDARY_NODE_MARKER, 1, &[0], Some(1.0), None, None),
            marker,
        );
        assert_invalid_graph(
            &decoder.add_edge(0, BOUNDARY_NODE_MARKER, &[0], Some(1.0), None, None),
            marker,
        );
        assert_invalid_graph(
            &decoder.add_edge_with_config(
                0,
                BOUNDARY_NODE_MARKER,
                &[0],
                crate::EdgeConfig::default(),
            ),
            marker,
        );
        assert_invalid_graph(
            &decoder.add_boundary_edge(BOUNDARY_NODE_MARKER, &[0], Some(1.0), None, None),
            marker,
        );
        assert_eq!((decoder.num_nodes(), decoder.num_edges()), (4, 0));

        assert_invalid_graph(
            &crate::PyMatchingDecoder::builder()
                .nodes(4)
                .add_edge(0, BOUNDARY_NODE_MARKER, vec![0], 1.0, None)
                .build(),
            marker,
        );
        assert_invalid_graph(
            &crate::PyMatchingDecoder::builder()
                .nodes(4)
                .add_boundary_edge(BOUNDARY_NODE_MARKER, vec![0], 1.0, None)
                .build(),
            marker,
        );
        let mut graph = petgraph::graph::UnGraph::new_undirected();
        let n0 = graph.add_node(crate::PyMatchingNode {
            id: 0,
            is_boundary: false,
        });
        let n1 = graph.add_node(crate::PyMatchingNode {
            id: BOUNDARY_NODE_MARKER,
            is_boundary: true,
        });
        graph.add_edge(
            n0,
            n1,
            crate::PyMatchingEdge {
                observables: vec![0],
                weight: 1.0,
                error_probability: None,
            },
        );
        assert_invalid_graph(
            &crate::pymatching_from_petgraph(&graph, &std::collections::HashSet::new(), 1),
            marker,
        );
    }

    #[test]
    fn observable_usize_max_is_rejected_before_the_edge_is_added() {
        let mut decoder = decoder_with_nodes(4);
        let message = format!("observable {} is out of range", usize::MAX);
        assert_invalid_graph(
            &decoder.add_edge(0, 1, &[0, usize::MAX], Some(1.0), None, None),
            &message,
        );
        assert_invalid_graph(
            &decoder.add_boundary_edge(0, &[usize::MAX], Some(1.0), None, None),
            &message,
        );
        assert_eq!(decoder.num_edges(), 0);
        assert_eq!(decoder.num_decoded_observables(), 1);
        assert_eq!(decoder.num_observables(), 64);
    }

    /// A graph whose observable count, `2**63 + 1`, no buffer can hold.
    fn unallocatable_observables_graph() -> crate::PyMatchingDecoder {
        let mut decoder = decoder_with_nodes(2);
        decoder
            .add_edge(0, 1, &[1 << 63], Some(1.0), None, None)
            .expect("upstream accepts a large observable index");
        decoder
    }

    #[test]
    fn unallocatable_observable_output_is_an_invalid_graph_on_every_entry_point() {
        let message = "sized by the graph's observable count";
        let syndrome = [1, 1];
        assert_invalid_graph(
            &unallocatable_observables_graph().decode(&syndrome),
            message,
        );
        for (bit_packed_input, bit_packed_output) in [(false, false), (true, true)] {
            let shots: &[u8] = if bit_packed_input { &[0b11] } else { &syndrome };
            let config = crate::BatchConfig {
                bit_packed_input,
                bit_packed_output,
                return_weights: true,
            };
            assert_invalid_graph(
                &unallocatable_observables_graph().decode_batch_with_config(shots, 1, 2, config),
                message,
            );
        }
        assert_invalid_graph(&unallocatable_observables_graph().add_noise(1, 7), message);
        assert_invalid_graph(
            &unallocatable_observables_graph().create_decode_buffer(),
            message,
        );
        // Matched pairs and edges have no observable-sized output, so they decode.
        let mut decoder = unallocatable_observables_graph();
        assert_eq!(
            decoder.decode_to_matched_pairs(&syndrome).expect("decodes"),
            [crate::MatchedPair {
                detector1: 0,
                detector2: Some(1)
            }]
        );
        assert_eq!(
            decoder.decode_to_edges(&syndrome).expect("decodes").len(),
            1
        );
    }

    #[test]
    fn output_too_large_only_for_the_shot_count_is_a_configuration_error() {
        // No detectors, so any shot count matches an empty shot buffer.
        let mut decoder = decoder_with_nodes(0);
        let result =
            decoder.decode_batch_with_config(&[], 1 << 62, 0, crate::BatchConfig::default());
        assert!(
            matches!(&result, Err(crate::PyMatchingError::Configuration(message))
                if message.contains("pass a smaller num_shots")),
            "{result:?}"
        );
        let result = decoder.add_noise(1 << 62, 7);
        assert!(
            matches!(&result, Err(crate::PyMatchingError::Configuration(message))
                if message.contains("pass a smaller num_samples")),
            "{result:?}"
        );
    }

    #[test]
    fn shortest_path_matcher_build_rejection_is_an_invalid_graph() {
        assert_matcher_build_rejected(&overweight_graph(1).get_shortest_path(0, 1));
    }

    /// A graph on `num_nodes` nodes with unit-weight `edges` and `boundary` nodes.
    fn path_graph(
        num_nodes: usize,
        edges: &[(usize, usize)],
        boundary: &[usize],
    ) -> crate::PyMatchingDecoder {
        let mut decoder = decoder_with_nodes(num_nodes);
        for &(node1, node2) in edges {
            decoder
                .add_edge(node1, node2, &[], Some(1.0), None, None)
                .expect("the edge is added");
        }
        decoder.set_boundary(boundary).expect("the boundary is set");
        decoder
    }

    #[test]
    fn shortest_path_cases_upstream_cannot_answer_are_configuration_errors() {
        // Each case dereferences a null pointer or throws in upstream.
        let cases: [(&str, crate::PyMatchingDecoder, usize, usize); 5] = [
            (
                "different connected components",
                path_graph(4, &[(0, 1), (2, 3)], &[]),
                0,
                3,
            ),
            (
                "different connected components",
                path_graph(3, &[(0, 1)], &[]),
                2,
                0,
            ),
            // Upstream never searches through a boundary node.
            (
                "different connected components",
                path_graph(3, &[(0, 1), (1, 2)], &[1]),
                0,
                2,
            ),
            (
                "no boundary is reachable from node 0",
                path_graph(4, &[(0, 1), (2, 3)], &[3]),
                0,
                3,
            ),
            (
                "are both boundary nodes",
                path_graph(3, &[(0, 1), (1, 2)], &[0, 2]),
                0,
                2,
            ),
        ];
        for (needle, mut decoder, source, target) in cases {
            let result = decoder.get_shortest_path(source, target);
            assert!(
                matches!(&result, Err(crate::PyMatchingError::Configuration(message))
                    if message.contains(needle)),
                "{source} -> {target}: {result:?}"
            );
        }
        let result = path_graph(2, &[], &[]).get_shortest_path(0, 1);
        assert!(
            matches!(&result, Err(crate::PyMatchingError::Configuration(message))
                if message.contains("empty graph")),
            "{result:?}"
        );
    }

    #[test]
    fn shortest_path_returns_upstream_paths() {
        let mut decoder = path_graph(3, &[(0, 1), (1, 2)], &[]);
        assert_eq!(
            decoder.get_shortest_path(0, 2).expect("connected"),
            [0, 1, 2]
        );
        assert_eq!(decoder.get_shortest_path(1, 1).expect("trivial"), [1]);
        // A boundary endpoint is left out; the path runs to the nearest boundary,
        // here node 1's boundary edge rather than boundary node 3.
        let mut decoder = path_graph(4, &[(0, 1)], &[3]);
        decoder
            .add_boundary_edge(1, &[], Some(1.0), None, None)
            .expect("the boundary edge is added");
        assert_eq!(
            decoder.get_shortest_path(0, 3).expect("boundary reachable"),
            [0, 1]
        );
        assert_eq!(
            decoder.get_shortest_path(3, 0).expect("boundary reachable"),
            [1, 0]
        );
    }

    #[test]
    fn shortest_path_uses_weights_set_after_an_earlier_query() {
        // 0 - 1 - 2 costs 2 against 10 for the direct edge, until every edge
        // gets the same weight.
        let mut decoder = path_graph(3, &[(0, 1), (1, 2)], &[]);
        decoder
            .add_edge(0, 2, &[], Some(10.0), None, None)
            .expect("the edge is added");
        assert_eq!(
            decoder.get_shortest_path(0, 2).expect("connected"),
            [0, 1, 2]
        );
        decoder
            .set_all_error_probabilities(0.1)
            .expect("the probability is valid");
        assert_eq!(decoder.get_shortest_path(0, 2).expect("connected"), [0, 2]);
    }

    #[test]
    fn check_matrix_caller_counts_are_configuration_errors() {
        let matrix = CheckMatrix::new(2, 3, vec![0, 0, 1, 1], vec![0, 1, 1, 2]);
        let build = |config: CheckMatrixConfig| {
            PyMatchingDecoder::from_check_matrix_with_config(&matrix, config).err()
        };
        let cases = [
            (
                CheckMatrixConfig {
                    repetitions: 0,
                    use_virtual_boundary: false,
                    ..Default::default()
                },
                "repetitions must be at least 1, got 0".to_string(),
            ),
            (
                CheckMatrixConfig {
                    repetitions: 1 << 63,
                    ..Default::default()
                },
                format!(
                    "2 rows times {} repetitions overflows usize; pass fewer repetitions",
                    1_usize << 63
                ),
            ),
            (
                CheckMatrixConfig {
                    error_probabilities: Some(vec![0.1]),
                    ..Default::default()
                },
                "error_probabilities has length 1 but must equal the number of columns (3)"
                    .to_string(),
            ),
        ];
        for (config, expected) in cases {
            let error = build(config);
            assert!(
                matches!(&error, Some(PyMatchingError::Configuration(message)) if *message == expected),
                "{error:?}"
            );
        }
        let error =
            PyMatchingDecoder::from_check_matrix(&CheckMatrix::new(1, 1 << 62, vec![], vec![]))
                .err();
        assert!(
            matches!(&error, Some(PyMatchingError::Configuration(message))
                if message.starts_with("cannot allocate one entry per column")),
            "{error:?}"
        );
    }

    #[test]
    fn check_matrix_without_rows_does_not_loop_over_repetitions() {
        // Zero rows pass the product check for any count, so a loop over the
        // empty rounds would not finish.
        let matrix = CheckMatrix::new(0, 1, vec![], vec![]);
        let config = CheckMatrixConfig {
            repetitions: 1 << 62,
            ..Default::default()
        };
        let decoder = PyMatchingDecoder::from_check_matrix_with_config(&matrix, config)
            .expect("a matrix without rows builds an empty graph");
        assert_eq!(decoder.num_nodes(), 0);
    }

    #[test]
    fn add_noise_syndrome_covers_nodes_past_the_detector_count() {
        // Boundary node 0 leaves 3 detectors, but the edge flips nodes 2 and 3.
        let mut decoder = decoder_with_nodes(4);
        decoder.set_boundary(&[0]).expect("the boundary is set");
        decoder
            .add_edge(2, 3, &[0], None, Some(0.5), None)
            .expect("the edge is added");
        decoder
            .add_edge(0, 1, &[], None, Some(0.5), None)
            .expect("the edge is added");
        assert_eq!((decoder.num_nodes(), decoder.num_detectors()), (4, 3));
        let noise = decoder.add_noise(64, 7).expect("noise is sampled");
        let mut flipped = 0;
        for (errors, syndrome) in noise.errors.iter().zip(&noise.syndromes) {
            assert_eq!(syndrome.len(), 4);
            assert_eq!(syndrome[0], 0, "a boundary node never fires");
            assert_eq!(syndrome[2], errors[0]);
            assert_eq!(syndrome[3], errors[0]);
            flipped += usize::from(errors[0]);
            decoder
                .decode(syndrome)
                .expect("the syndrome length is accepted");
        }
        assert!(0 < flipped && flipped < 64, "{flipped}");
    }

    #[test]
    fn disallowed_parallel_edge_is_an_invalid_graph() {
        let mut decoder =
            crate::PyMatchingDecoder::from_dem("error(0.1) D0 D1").expect("valid DEM builds");
        let error = decoder
            .add_edge(
                0,
                1,
                &[],
                Some(1.0),
                None,
                Some(crate::MergeStrategy::Disallow),
            )
            .expect_err("a disallowed parallel edge is rejected");
        assert!(
            matches!(&error, crate::PyMatchingError::InvalidGraph(message)
                if message.starts_with("Failed to add edge: ")
                    && message.contains("already exists")),
            "{error:?}"
        );
        decoder
            .add_boundary_edge(0, &[], Some(1.0), None, None)
            .expect("the first boundary edge is added");
        let error = decoder
            .add_boundary_edge(
                0,
                &[],
                Some(1.0),
                None,
                Some(crate::MergeStrategy::Disallow),
            )
            .expect_err("a disallowed parallel boundary edge is rejected");
        assert!(
            matches!(&error, crate::PyMatchingError::InvalidGraph(message)
                if message.starts_with("Failed to add boundary edge: ")
                    && message.contains("already exists")),
            "{error:?}"
        );
    }

    #[test]
    fn correlated_prior_above_half_is_a_configuration_error() {
        let dem = "error(0.7) D0 L0\nerror(0.1) D0 D1";
        let result = crate::PyMatchingDecoder::from_dem_with_correlations(dem, true);
        assert!(
            matches!(
                &result,
                Err(crate::PyMatchingError::Configuration(message))
                    if message.starts_with("DEM error probability 0.7 is above 0.5")
            ),
            "{:?}",
            result.err()
        );
        assert!(crate::PyMatchingDecoder::from_dem_with_correlations(dem, false).is_ok());
    }

    #[test]
    fn observable_batch_rejects_short_stride() {
        let mut decoder =
            crate::PyMatchingDecoder::from_dem("error(0.1) D0 L0\ndetector D2").unwrap();
        assert!(matches!(
            pecos_decoder_core::ObservableDecoder::decode_batch_to_observables(
                &mut decoder,
                &[0],
                1,
                1
            ),
            Err(pecos_decoder_core::DecoderError::InvalidDimensions {
                expected: 3,
                actual: 1
            })
        ));
    }

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
        assert!(
            !result.observable.is_empty(),
            "decoder returned no observables"
        );
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

    #[test]
    fn native_observable_batch_is_correct_at_64_and_65_observables() {
        for observable in [63, 64] {
            let dem = format!("error(0.1) D0 L{observable}\n");
            let mut decoder = PyMatchingDecoder::from_dem(&dem).unwrap();
            let masks = decoder
                .decode_batch_to_observables(&[1, 0, 1], 3, 1)
                .unwrap();
            assert_eq!(masks.len(), 3);
            assert!(masks[0].get(observable));
            assert!(masks[1].is_zero());
            assert!(masks[2].get(observable));
            assert_eq!(masks[0], decoder.decode_obs(&[1]).unwrap());
        }
    }

    #[test]
    fn native_observable_batch_without_observables_preserves_shots() {
        for correlated in [false, true] {
            let mut decoder =
                PyMatchingDecoder::from_dem_with_correlations("error(0.1) D0 D1", correlated)
                    .unwrap();
            let masks = decoder
                .decode_batch_to_observables(&[0, 0, 1, 1], 2, 2)
                .unwrap();
            assert_eq!(masks.len(), 2);
            for (mask, syndrome) in masks.iter().zip([[0, 0], [1, 1]]) {
                assert!(mask.is_zero());
                assert_eq!(*mask, decoder.decode_obs(&syndrome).unwrap());
            }
        }
    }

    #[test]
    fn correlated_mode_changes_the_correlation_sensitive_fixture() {
        let dem = "error(0.01) D0 D1 ^ D2 D3 L0\nerror(0.1) D2\nerror(0.1) D3\n";
        let mut correlated = PyMatchingDecoder::from_dem_with_correlations(dem, true).unwrap();
        let mut uncorrelated = PyMatchingDecoder::from_dem_with_correlations(dem, false).unwrap();

        assert_eq!(
            correlated.decode_obs(&[1, 1, 1, 1]).unwrap().to_u64(),
            Some(1)
        );
        assert_eq!(
            uncorrelated.decode_obs(&[1, 1, 1, 1]).unwrap().to_u64(),
            Some(0)
        );
    }

    #[test]
    fn test_14_correlated_edge_decode() {
        let dem = "error(0.01) D0 D1 ^ D2 D3 L0\nerror(0.1) D2\nerror(0.1) D3\n";
        let syndrome = [1, 1, 1, 1];
        let selections = [
            (false, vec![(0, Some(1)), (2, None), (3, None)]),
            (true, vec![(0, Some(1)), (2, Some(3))]),
        ];
        for (correlated, expected_edges) in selections {
            let mut decoder =
                PyMatchingDecoder::from_dem_with_correlations(dem, correlated).unwrap();
            let expected_obs = decoder.decode_to_observables(&syndrome).unwrap();
            let mut selected: Vec<_> = decoder
                .decode_to_edges(&syndrome)
                .unwrap()
                .into_iter()
                .map(|pair| match pair.detector2 {
                    Some(b) if b < pair.detector1 => (b, Some(pair.detector1)),
                    b => (pair.detector1, b),
                })
                .collect();
            selected.sort_unstable();
            assert_eq!(
                selected, expected_edges,
                "edge selection must honor correlated={correlated}"
            );
            let selected_obs = u64::from(selected.contains(&(2, Some(3))));
            assert_eq!(
                selected_obs, expected_obs,
                "edge observable must match monolithic decoding"
            );
            let detailed =
                DetailedDecoder::decode_to_edges(&mut decoder, &ArrayView1::from(&syndrome))
                    .unwrap();
            let detailed_obs = detailed
                .iter()
                .flat_map(|edge| &edge.observables)
                .fold(0u64, |mask, &observable| mask ^ (1u64 << observable));
            assert_eq!(
                detailed_obs, expected_obs,
                "detailed edge decode must use the same correlated correction"
            );
        }
    }

    #[test]
    fn detailed_edges_include_intermediate_detectors() {
        let dem = "error(0.1) D0 D1 L0\nerror(0.1) D1 D2\nerror(0.001) D0\n";
        let mut decoder = PyMatchingDecoder::from_dem(dem).unwrap();
        let syndrome = [1, 0, 1];
        let expected = decoder.decode_to_observables(&syndrome).unwrap();
        let edges =
            DetailedDecoder::decode_to_edges(&mut decoder, &ArrayView1::from(&syndrome)).unwrap();
        let mut incidence = [0u8; 3];
        let mut observable = 0;
        for edge in &edges {
            incidence[edge.node1] ^= 1;
            if edge.node2 < incidence.len() {
                incidence[edge.node2] ^= 1;
            }
            for &o in &edge.observables {
                observable ^= 1u64 << o;
            }
        }
        assert_eq!(
            edges.len(),
            2,
            "edge decode must report the path through detector 1"
        );
        assert_eq!(incidence, syndrome);
        assert_eq!(observable, expected);
    }
}
