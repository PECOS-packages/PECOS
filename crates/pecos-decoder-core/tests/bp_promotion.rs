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

use pecos_decoder_core::bp::{BpGraph, BpScratch, min_sum_bp_into};
use pecos_decoder_core::dem::{DemCheckMatrix, SparseDem};
use pecos_decoder_core::errors::DecoderError;
use std::collections::BTreeMap;

const DEM: &str = "\
error(0.03) D0 D1 L0
error(0.17) D1 D2
error(0.41) D0 D2 L1
error(0.08) D2 D3
error(0.29) D3
error(0.5) D0 D1 D3
error(0.0) L2
";

struct PromotionFixture {
    syndrome: [u8; 4],
    iterations: usize,
    serial: bool,
    posterior_bits: [u64; 7],
}

// Captured from pecos-uf-decoder/src/mini_bp.rs at a7af1e588 before the
// implementation moved crates. The cases pin zero-iteration, flooding,
// serial, and the >=6-iteration EWAInit paths.
const PRE_PROMOTION_FIXTURES: [PromotionFixture; 6] = [
    PromotionFixture {
        syndrome: [0, 0, 0, 0],
        iterations: 0,
        serial: false,
        posterior_bits: [
            4_615_009_897_182_140_011,
            4_609_819_849_526_776_594,
            4_600_228_237_266_466_083,
            4_612_682_095_399_216_500,
            4_606_240_122_066_615_758,
            0,
            4_629_137_466_983_448_576,
        ],
    },
    PromotionFixture {
        syndrome: [1, 0, 1, 0],
        iterations: 1,
        serial: false,
        posterior_bits: [
            4_615_009_897_182_140_011,
            4_608_795_378_066_064_068,
            13_827_195_235_668_434_244,
            4_612_169_859_668_860_237,
            4_606_240_122_066_615_758,
            4_608_637_773_721_183_982,
            4_629_137_466_983_448_576,
        ],
    },
    PromotionFixture {
        syndrome: [1, 0, 1, 0],
        iterations: 5,
        serial: false,
        posterior_bits: [
            4_616_824_137_156_046_395,
            4_612_772_222_837_049_545,
            13_834_538_819_755_652_347,
            4_614_632_360_329_047_982,
            4_610_536_441_559_920_256,
            4_612_079_460_430_909_568,
            4_629_137_466_983_448_576,
        ],
    },
    PromotionFixture {
        syndrome: [1, 0, 1, 0],
        iterations: 5,
        serial: true,
        posterior_bits: [
            4_617_146_141_481_849_060,
            4_613_378_948_369_012_257,
            13_835_398_173_110_397_208,
            4_614_908_232_143_345_217,
            4_611_780_850_137_401_864,
            4_612_614_139_529_345_710,
            4_629_137_466_983_448_576,
        ],
    },
    PromotionFixture {
        syndrome: [1, 1, 0, 1],
        iterations: 6,
        serial: false,
        posterior_bits: [
            4_618_266_947_953_514_113,
            4_615_706_092_554_912_435,
            4_615_101_373_839_185_582,
            4_616_296_009_642_864_701,
            4_612_933_259_521_655_288,
            13_838_383_818_924_061_210,
            4_629_137_466_983_448_576,
        ],
    },
    PromotionFixture {
        syndrome: [1, 1, 0, 1],
        iterations: 7,
        serial: true,
        posterior_bits: [
            4_618_452_438_281_817_700,
            4_616_137_273_858_435_077,
            4_615_361_769_219_560_688,
            4_616_412_249_689_813_738,
            4_613_356_255_446_979_550,
            13_838_660_022_657_093_308,
            4_629_137_466_983_448_576,
        ],
    },
];

#[test]
fn posteriors_match_the_pre_promotion_bit_fixture() {
    let dcm = DemCheckMatrix::from_dem_str(DEM).unwrap();
    let graph = BpGraph::from_dcm(&dcm);
    let mut scratch = BpScratch::new(&graph);
    let mut posterior = vec![0.0; graph.mechanism_count()];

    for fixture in PRE_PROMOTION_FIXTURES {
        min_sum_bp_into(
            &graph,
            &fixture.syndrome,
            fixture.iterations,
            0.625,
            fixture.serial,
            &mut scratch,
            &mut posterior,
        )
        .unwrap();
        assert_eq!(
            posterior
                .iter()
                .copied()
                .map(f64::to_bits)
                .collect::<Vec<_>>(),
            fixture.posterior_bits,
            "syndrome={:?}, iterations={}, serial={}",
            fixture.syndrome,
            fixture.iterations,
            fixture.serial
        );
    }
}

#[test]
fn sparse_and_dense_graph_construction_agree() {
    let dcm = DemCheckMatrix::from_dem_str(DEM).unwrap();
    let sparse = SparseDem::from_dem_str(DEM).unwrap();
    let dense_graph = BpGraph::from_dcm(&dcm);
    let sparse_graph = BpGraph::from_sparse_dem(&sparse).unwrap();

    assert_eq!(dense_graph.check_count(), sparse_graph.check_count());
    assert_eq!(
        dense_graph.mechanism_count(),
        sparse_graph.mechanism_count()
    );
    assert_eq!(dense_graph.edge_count(), sparse_graph.edge_count());
    assert_eq!(
        dense_graph
            .prior_llrs()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        sparse_graph
            .prior_llrs()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    );
}

#[test]
fn rejects_short_and_long_syndromes() {
    let graph = BpGraph::from_dcm(&DemCheckMatrix::from_dem_str(DEM).unwrap());
    let mut scratch = BpScratch::new(&graph);
    let mut posterior = vec![0.0; graph.mechanism_count()];

    for actual in [graph.check_count() - 1, graph.check_count() + 1] {
        let error = min_sum_bp_into(
            &graph,
            &vec![0; actual],
            3,
            0.625,
            true,
            &mut scratch,
            &mut posterior,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            DecoderError::InvalidDimensions {
                expected: 4,
                actual: error_actual
            } if error_actual == actual
        ));
    }
}

#[test]
fn sparse_constructor_rejects_invalid_public_models() {
    let duplicate = SparseDem {
        mechanisms: vec![(0.1, vec![0, 0], vec![])],
        detector_coords: BTreeMap::new(),
        num_detectors: 1,
        num_observables: 0,
    };
    assert!(BpGraph::from_sparse_dem(&duplicate).is_err());

    let out_of_range = SparseDem {
        mechanisms: vec![(0.1, vec![1], vec![])],
        detector_coords: BTreeMap::new(),
        num_detectors: 1,
        num_observables: 0,
    };
    assert!(BpGraph::from_sparse_dem(&out_of_range).is_err());
}
