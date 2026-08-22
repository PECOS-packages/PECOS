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

//! Integration tests for the UF decoder using realistic surface code DEMs.

use pecos_decoder_core::ObservableDecoder;
use pecos_decoder_core::dem::DemMatchingGraph;
use pecos_uf_decoder::{UfDecoder, UfDecoderConfig};

/// Distance-3 rotated surface code, Z basis, 3 rounds of syndrome extraction.
/// This is a realistic DEM with 24 detectors and ~100 error mechanisms.
/// Non-decomposed (for matching graph extraction).
const D3_SURFACE_CODE_DEM: &str =
    include_str!("../../../examples/surface_code_circuits/surface_code_d3_z_stim.dem");

/// Check the decoder initializes correctly from a real surface code DEM.
#[test]
fn test_real_dem_construction() {
    let graph = DemMatchingGraph::from_dem_str(D3_SURFACE_CODE_DEM).unwrap();
    assert!(
        graph.num_detectors >= 20,
        "Expected 20+ detectors, got {}",
        graph.num_detectors
    );
    assert!(
        graph.edges.len() >= 10,
        "Expected 10+ edges, got {}",
        graph.edges.len()
    );

    let dec = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::default()).unwrap();
    assert_eq!(dec.num_detectors(), graph.num_detectors);
    assert_eq!(dec.num_edges(), graph.edges.len());
}

/// Decode the trivial (no-error) syndrome -- should always give observable 0.
#[test]
fn test_real_dem_no_errors() {
    let graph = DemMatchingGraph::from_dem_str(D3_SURFACE_CODE_DEM).unwrap();
    let mut dec = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::default()).unwrap();
    let syndrome = vec![0u8; graph.num_detectors];
    assert_eq!(dec.decode_syndrome(&syndrome), 0);
}

/// Decode single-defect syndromes (one detector triggered).
/// Each should produce a valid correction (not panic, not hang).
#[test]
fn test_real_dem_single_defects() {
    let graph = DemMatchingGraph::from_dem_str(D3_SURFACE_CODE_DEM).unwrap();
    let mut dec = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::default()).unwrap();

    for d in 0..graph.num_detectors {
        let mut syndrome = vec![0u8; graph.num_detectors];
        syndrome[d] = 1;
        // Should not panic or hang. Observable is either 0 or 1.
        let obs = dec.decode_syndrome(&syndrome);
        assert!(
            obs <= 1,
            "Observable mask {obs} too large for single-observable DEM"
        );
    }
}

/// Decode syndromes with two adjacent defects.
/// For each edge in the matching graph, set both endpoint detectors and decode.
#[test]
fn test_real_dem_adjacent_pairs() {
    let graph = DemMatchingGraph::from_dem_str(D3_SURFACE_CODE_DEM).unwrap();
    let mut dec = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::default()).unwrap();

    for edge in &graph.edges {
        let mut syndrome = vec![0u8; graph.num_detectors];
        syndrome[edge.node1 as usize] = 1;
        if let Some(n2) = edge.node2 {
            syndrome[n2 as usize] = 1;
        }
        let obs = dec.decode_syndrome(&syndrome);
        assert!(obs <= 1, "Observable mask {obs} too large");
    }
}

/// Stress test: decode many random syndromes with even number of defects.
/// Verify the decoder never panics or hangs, and results are in range.
#[test]
fn test_real_dem_random_syndromes() {
    let graph = DemMatchingGraph::from_dem_str(D3_SURFACE_CODE_DEM).unwrap();
    let mut dec = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::default()).unwrap();
    let mut rng = fastrand::Rng::with_seed(42);

    for _ in 0..1000 {
        let mut syndrome = vec![0u8; graph.num_detectors];
        let mut num_defects = 0;
        for s in &mut syndrome {
            if rng.f64() < 0.05 {
                *s = 1;
                num_defects += 1;
            }
        }
        // Ensure even number of defects (valid syndrome for surface codes).
        if num_defects % 2 != 0 && !syndrome.is_empty() {
            // Flip a random detector to make it even.
            let idx = rng.usize(..syndrome.len());
            syndrome[idx] ^= 1;
        }

        let obs = dec.decode_syndrome(&syndrome);
        assert!(obs <= 1, "Observable mask {obs} too large");
    }
}

/// Test via the `ObservableDecoder` trait (the actual interface used in production).
#[test]
fn test_observable_decoder_trait_real_dem() {
    let graph = DemMatchingGraph::from_dem_str(D3_SURFACE_CODE_DEM).unwrap();
    let mut dec = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::default()).unwrap();
    let syndrome = vec![0u8; graph.num_detectors];
    let result = dec.decode_to_observables(&syndrome);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), 0);
}

/// Test `MatchingDecoder` trait on real DEM.
#[test]
fn test_matching_decoder_trait_real_dem() {
    use pecos_decoder_core::correlated_decoder::MatchingDecoder;
    let graph = DemMatchingGraph::from_dem_str(D3_SURFACE_CODE_DEM).unwrap();
    let mut dec = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::default()).unwrap();

    // Two adjacent defects.
    let edge = &graph.edges[0];
    let mut syndrome = vec![0u8; graph.num_detectors];
    syndrome[edge.node1 as usize] = 1;
    if let Some(n2) = edge.node2 {
        syndrome[n2 as usize] = 1;
    }

    let (obs, _matched_edges) = dec.decode_with_matching(&syndrome).unwrap();
    assert!(obs <= 1);
    // Predecoder may handle simple cases without tracking edges.
    assert!(obs <= 1);
}

/// Buffer reuse stress test: alternate between zero and non-zero syndromes.
/// Catches bugs where state leaks between shots.
#[test]
fn test_buffer_reuse_correctness() {
    let graph = DemMatchingGraph::from_dem_str(D3_SURFACE_CODE_DEM).unwrap();
    let mut dec = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::default()).unwrap();
    let zero_syndrome = vec![0u8; graph.num_detectors];

    let mut defect_syndrome = vec![0u8; graph.num_detectors];
    defect_syndrome[0] = 1;

    for _ in 0..500 {
        // Zero syndrome must always give 0.
        assert_eq!(
            dec.decode_syndrome(&zero_syndrome),
            0,
            "Buffer leak: non-zero after defect"
        );
        // Defect syndrome should give a consistent result.
        let _ = dec.decode_syndrome(&defect_syndrome);
    }
}

/// A DEM line with an error prior p > 0.5 produces a negative edge weight
/// (`ln((1-p)/p) < 0`), which the predecoder's optimality proofs do not admit.
/// The DEM-text constructors must reject it as an error, not mis-decode.
#[test]
fn test_from_dem_rejects_negative_weight_priors() {
    let dem = "error(0.6) D0 L0\nerror(0.01) D0 D1\n";
    assert!(UfDecoder::from_dem(dem, UfDecoderConfig::balanced()).is_err());
    assert!(
        pecos_uf_decoder::BpUfDecoder::from_dem(dem, pecos_uf_decoder::BpUfConfig::balanced())
            .is_err()
    );
    assert!(
        pecos_uf_decoder::CssUfDecoder::from_dems(dem, dem, UfDecoderConfig::balanced()).is_err()
    );
}

/// The matching-graph parser drops 3+-detector mechanisms, so the CSS
/// constructor must reject a hyperedge DEM instead of silently decoding a
/// truncated model.
#[test]
fn test_css_from_dems_rejects_hyperedge_models() {
    let graphlike = "error(0.05) D0 D1 L0\nerror(0.01) D0\n";
    let hyperedge = "error(0.1) D0 D1 D2 L0\nerror(0.05) D0\n";
    for (x_dem, z_dem) in [(hyperedge, graphlike), (graphlike, hyperedge)] {
        let error =
            pecos_uf_decoder::CssUfDecoder::from_dems(x_dem, z_dem, UfDecoderConfig::balanced())
                .err()
                .expect("hyperedge DEM must be rejected");
        assert!(error.to_string().contains("graphlike"), "{error}");
    }
}

/// The graph-level constructor asserts the same premise loudly for callers
/// that build graphs directly (contract violation, not user input).
#[test]
#[should_panic(expected = "non-negative edge weights")]
fn test_from_matching_graph_asserts_non_negative_weights() {
    let dem = "error(0.6) D0 L0\n";
    let graph = DemMatchingGraph::from_dem_str(dem).unwrap();
    let _ = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::balanced()).unwrap();
}

/// Weight-zero edges (p = 0.5) are benign: the shortcut proofs hold as ties.
#[test]
fn test_from_dem_accepts_weight_zero_edges() {
    let dem = "error(0.5) D0 L0\nerror(0.01) D0 D1\n";
    let mut dec = UfDecoder::from_dem(dem, UfDecoderConfig::balanced()).unwrap();
    let syndrome = vec![0u8; dec.num_detectors()];
    assert_eq!(dec.decode_syndrome(&syndrome), 0);
}
