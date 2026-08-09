// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

use pecos_decoder_core::ObservableDecoder;
use pecos_frontier::{
    BpTrellisConfig, BpTrellisDecoder, DecoderError, FrontierConfig, FrontierDecoder,
    FrontierResult, SparseDem, TrellisOrdering, backward_deadline_column_order,
    deadline_column_order,
};
use std::collections::BTreeMap;

fn sparse_dem(
    mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    num_detectors: usize,
    num_observables: usize,
) -> SparseDem {
    SparseDem {
        mechanisms,
        detector_coords: BTreeMap::new(),
        num_detectors,
        num_observables,
    }
}

fn assert_results_bitwise_equal_except_timing(left: &FrontierResult, right: &FrontierResult) {
    assert!(left.bp_seconds > 0.0);
    assert!(right.bp_seconds > 0.0);
    let mut left = left.clone();
    let mut right = right.clone();
    left.bp_seconds = 0.0;
    right.bp_seconds = 0.0;
    assert_eq!(left, right);
}

fn defaults_identity_dem() -> SparseDem {
    let mut mechanisms = Vec::new();
    for detector in 0..5 {
        mechanisms.push((
            0.10 + 0.01 * f64::from(detector),
            vec![detector],
            vec![detector + 4],
        ));
    }
    for observable in 0..4 {
        mechanisms.push((
            0.20 + 0.01 * f64::from(observable),
            vec![],
            vec![observable],
        ));
    }
    for detector in 0..5 {
        mechanisms.push((0.15 + 0.01 * f64::from(detector), vec![detector], vec![]));
    }
    // This planted pair is merged exactly by the BpTrellis default.
    mechanisms.push((0.07, vec![5], vec![]));
    mechanisms.push((0.07, vec![5], vec![]));
    sparse_dem(mechanisms, 6, 9)
}

#[test]
fn bptrellis_defaults_enable_bp_merge_and_deadline_order() {
    let defaults = BpTrellisConfig::default();
    assert_eq!(defaults.k, 8);
    assert_eq!(defaults.delta.to_bits(), 100.0_f64.to_bits());
    assert_eq!(defaults.score_alpha.to_bits(), 0.8_f64.to_bits());
    assert_eq!(defaults.bp_score_iterations, 5);
    assert!(defaults.merge_indistinguishable);
    assert_eq!(defaults.ordering, TrellisOrdering::Deadline);

    let dem = defaults_identity_dem();
    let deadline_order = deadline_column_order(&dem).unwrap();
    assert_ne!(
        deadline_order,
        (0..dem.mechanisms.len()).collect::<Vec<_>>()
    );

    let mut decoder = BpTrellisDecoder::from_sparse_dem(&dem, defaults).unwrap();
    let mut explicit_deadline = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            k: 8,
            delta: 100.0,
            score_alpha: 0.8,
            column_order: Some(deadline_order),
            merge_indistinguishable: true,
            bp_score_iterations: 5,
        },
    )
    .unwrap();
    let mut time_order = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            k: 8,
            delta: 100.0,
            score_alpha: 0.8,
            column_order: None,
            merge_indistinguishable: true,
            bp_score_iterations: 5,
        },
    )
    .unwrap();

    let syndrome = [0; 6];
    let actual = decoder.decode(&syndrome).unwrap();
    let expected = explicit_deadline.decode(&syndrome).unwrap();
    let time_order_result = time_order.decode(&syndrome).unwrap();

    assert_eq!(actual.processed_columns, dem.mechanisms.len() - 1);
    assert!(actual.dropped_states > 0, "the finite default K must prune");
    assert_results_bitwise_equal_except_timing(&actual, &expected);
    assert_ne!(
        actual.dropped_log_mass.to_bits(),
        time_order_result.dropped_log_mass.to_bits(),
        "this fixture must distinguish deadline order from DEM order"
    );
    assert!(decoder.build_seconds() >= 0.0);
}

#[test]
fn bptrellis_matches_hand_mapped_frontier_for_every_ordering() {
    let dem = sparse_dem(
        vec![
            (0.12, vec![0], vec![0]),
            (0.08, vec![1], vec![]),
            (0.15, vec![0, 1], vec![1]),
            (0.12, vec![0], vec![0]),
            (0.05, vec![], vec![0]),
        ],
        2,
        2,
    );
    let cases = [
        (
            TrellisOrdering::Deadline,
            Some(deadline_column_order(&dem).unwrap()),
        ),
        (
            TrellisOrdering::BackwardDeadline,
            Some(backward_deadline_column_order(&dem).unwrap()),
        ),
        (TrellisOrdering::TimeOrder, None),
        (
            TrellisOrdering::Explicit(vec![4, 2, 1, 3, 0]),
            Some(vec![4, 2, 1, 3, 0]),
        ),
    ];

    for (ordering, column_order) in cases {
        let mut decoder = BpTrellisDecoder::from_sparse_dem(
            &dem,
            BpTrellisConfig {
                k: 4,
                delta: 12.0,
                score_alpha: 0.6,
                bp_score_iterations: 0,
                merge_indistinguishable: true,
                ordering,
            },
        )
        .unwrap();
        let mut frontier = FrontierDecoder::from_sparse_dem(
            &dem,
            FrontierConfig {
                k: 4,
                delta: 12.0,
                score_alpha: 0.6,
                column_order,
                merge_indistinguishable: true,
                bp_score_iterations: 0,
            },
        )
        .unwrap();

        for syndrome in [[0, 0], [1, 0], [0, 1], [1, 1]] {
            assert_eq!(
                decoder.decode(&syndrome).unwrap(),
                frontier.decode(&syndrome).unwrap()
            );
        }
    }
}

#[test]
fn bptrellis_works_through_observable_decoder_trait_object() {
    let decoder =
        BpTrellisDecoder::from_dem_str("error(0.2) D0 L0\n", BpTrellisConfig::default()).unwrap();
    let mut boxed: Box<dyn ObservableDecoder> = Box::new(decoder);

    assert_eq!(boxed.decode_to_observables(&[1]).unwrap(), 1);
}

#[test]
fn bptrellis_explicit_order_validation_propagates() {
    let dem = sparse_dem(vec![(0.1, vec![0], vec![]), (0.2, vec![0], vec![])], 1, 0);
    let error = BpTrellisDecoder::from_sparse_dem(
        &dem,
        BpTrellisConfig {
            ordering: TrellisOrdering::Explicit(vec![0, 0]),
            ..BpTrellisConfig::default()
        },
    )
    .unwrap_err();

    assert!(matches!(error, DecoderError::InvalidConfiguration(_)));
    assert!(error.to_string().contains("permutation"));
}
