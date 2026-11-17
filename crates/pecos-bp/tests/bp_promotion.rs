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

use pecos_bp::{BpGraph, BpScratch, min_sum_bp_into};
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
// serial, and the >=6-iteration EWAInit paths. Refreshed for the native prior
// magnitude floor: the error(0.5) mechanism now starts at +4e-5, which changes
// its own zero-iteration belief and propagates through connected messages.
const FLOORED_PROMOTION_FIXTURES: [PromotionFixture; 6] = [
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
            4_541_027_782_865_676_529,
            4_629_137_466_983_448_576,
        ],
    },
    PromotionFixture {
        syndrome: [1, 0, 1, 0],
        iterations: 1,
        serial: false,
        posterior_bits: [
            4_615_009_897_182_140_011,
            4_608_795_490_656_054_753,
            13_827_195_460_848_415_614,
            4_612_169_915_963_855_579,
            4_606_240_347_246_597_127,
            4_608_637_953_865_169_077,
            4_629_137_466_983_448_576,
        ],
    },
    PromotionFixture {
        syndrome: [1, 0, 1, 0],
        iterations: 5,
        serial: false,
        posterior_bits: [
            4_616_824_180_813_744_714,
            4_612_772_327_043_747_251,
            13_834_539_028_169_047_759,
            4_614_632_398_093_890_110,
            4_610_536_594_737_351_886,
            4_612_079_592_249_984_231,
            4_629_137_466_983_448_576,
        ],
    },
    PromotionFixture {
        syndrome: [1, 0, 1, 0],
        iterations: 5,
        serial: true,
        posterior_bits: [
            4_617_146_191_502_256_494,
            4_613_379_065_134_390_077,
            13_835_398_289_199_190_492,
            4_614_908_275_977_268_528,
            4_611_780_938_972_821_018,
            4_612_614_282_371_173_107,
            4_629_137_466_983_448_576,
        ],
    },
    PromotionFixture {
        syndrome: [1, 1, 0, 1],
        iterations: 6,
        serial: false,
        posterior_bits: [
            4_618_266_892_009_420_954,
            4_615_705_952_908_721_583,
            4_615_101_234_192_994_729,
            4_616_295_981_315_152_887,
            4_612_933_162_219_654_380,
            13_838_383_673_256_253_731,
            4_629_137_466_983_448_576,
        ],
    },
    PromotionFixture {
        syndrome: [1, 1, 0, 1],
        iterations: 7,
        serial: true,
        posterior_bits: [
            4_618_452_376_631_751_592,
            4_616_137_131_422_347_610,
            4_615_361_628_431_883_320,
            4_616_412_220_205_433_308,
            4_613_356_155_224_562_520,
            13_838_659_856_612_447_261,
            4_629_137_466_983_448_576,
        ],
    },
];

#[test]
fn posteriors_match_the_floored_promotion_bit_fixture() {
    let dcm = DemCheckMatrix::from_dem_str(DEM).unwrap();
    let graph = BpGraph::from_dcm(&dcm);
    let mut scratch = BpScratch::new(&graph);
    let mut posterior = vec![0.0; graph.mechanism_count()];

    for fixture in FLOORED_PROMOTION_FIXTURES {
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
