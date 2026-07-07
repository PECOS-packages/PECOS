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

//! Cross-decoder comparison: UF vs itself (consistency checks).
//!
//! We can't depend on pecos-pymatching here (C++ build), but we CAN
//! verify that UF + ensemble composed of UF decoders all agree.

use pecos_decoder_core::ObservableDecoder;
use pecos_decoder_core::dem::DemMatchingGraph;
use pecos_decoder_core::ensemble::EnsembleDecoder;
use pecos_uf_decoder::{UfDecoder, UfDecoderConfig};

const D3_DEM: &str =
    include_str!("../../../examples/surface_code_circuits/surface_code_d3_z_stim.dem");

/// An ensemble of 3 identical UF decoders must agree with a single UF decoder
/// on every syndrome (since all members produce the same result, the majority
/// vote is identical to any single member).
#[test]
fn test_ensemble_of_identical_decoders_matches_single() {
    let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();

    let mut single = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::default());
    let members: Vec<Box<dyn ObservableDecoder>> = (0..3)
        .map(|_| {
            Box::new(UfDecoder::from_matching_graph(
                &graph,
                UfDecoderConfig::default(),
            )) as Box<dyn ObservableDecoder>
        })
        .collect();
    let mut ensemble = EnsembleDecoder::new(members);

    let mut rng = fastrand::Rng::with_seed(123);
    for _ in 0..500 {
        let mut syndrome = vec![0u8; graph.num_detectors];
        for s in &mut syndrome {
            if rng.f64() < 0.05 {
                *s = 1;
            }
        }

        let single_obs = single.decode_to_observables(&syndrome).unwrap();
        let ensemble_obs = ensemble.decode_to_observables(&syndrome).unwrap();
        assert_eq!(
            single_obs, ensemble_obs,
            "Ensemble of identical decoders diverged from single decoder"
        );
    }
}

/// Verify the UF decoder produces deterministic results across runs.
#[test]
fn test_deterministic_results() {
    let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();

    let mut dec1 = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::default());
    let mut dec2 = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::default());

    let mut rng = fastrand::Rng::with_seed(999);
    for _ in 0..200 {
        let mut syndrome = vec![0u8; graph.num_detectors];
        for s in &mut syndrome {
            if rng.f64() < 0.08 {
                *s = 1;
            }
        }

        let r1 = dec1.decode_to_observables(&syndrome).unwrap();
        let r2 = dec2.decode_to_observables(&syndrome).unwrap();
        assert_eq!(r1, r2, "Two identical decoders gave different results");
    }
}

/// Test `MatchingDecoder` consistency: `decode_with_matching` and `decode_to_observables`
/// must agree on the observable mask.
#[test]
fn test_matching_agrees_with_observable() {
    use pecos_decoder_core::correlated_decoder::MatchingDecoder;

    let graph = DemMatchingGraph::from_dem_str(D3_DEM).unwrap();
    let mut dec = UfDecoder::from_matching_graph(&graph, UfDecoderConfig::default());

    let mut rng = fastrand::Rng::with_seed(777);
    for _ in 0..200 {
        let mut syndrome = vec![0u8; graph.num_detectors];
        for s in &mut syndrome {
            if rng.f64() < 0.06 {
                *s = 1;
            }
        }

        let obs = dec.decode_to_observables(&syndrome).unwrap();

        // Reset and decode again with matching
        let (match_obs, edges) = dec.decode_with_matching(&syndrome).unwrap();

        assert_eq!(
            obs, match_obs,
            "ObservableDecoder and MatchingDecoder disagree on observable mask"
        );

        // If there are matched edges, they should be valid indices
        for &e in &edges {
            assert!(e < dec.num_edges(), "Edge index {e} out of range");
        }
    }
}
