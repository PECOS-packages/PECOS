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

use pecos_bp_trellis::{BpTrellisConfig, BpTrellisDecoder, TrellisOrdering};
use pecos_decoder_core::ObservableDecoder;
use pecos_trellis::{
    DecoderError, ObsMask, SparseDem, TrellisConfig, TrellisDecoder, TrellisResult,
    backward_deadline_column_order, deadline_column_order,
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

fn assert_results_bitwise_equal_except_timing(left: &TrellisResult, right: &TrellisResult) {
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
    assert!(defaults.escalation_ks.is_empty());

    let dem = defaults_identity_dem();
    let deadline_order = deadline_column_order(&dem).unwrap();
    assert_ne!(
        deadline_order,
        (0..dem.mechanisms.len()).collect::<Vec<_>>()
    );

    let mut decoder = BpTrellisDecoder::from_sparse_dem(&dem, defaults).unwrap();
    let mut explicit_deadline = TrellisDecoder::from_sparse_dem(
        &dem,
        TrellisConfig {
            k: 8,
            delta: 100.0,
            score_alpha: 0.8,
            column_order: Some(deadline_order),
            merge_indistinguishable: true,
            bp_score_iterations: 5,
            metric_mode: pecos_trellis::MetricMode::default(),
            int_metric_scale: 1024,
        },
    )
    .unwrap();
    let mut time_order = TrellisDecoder::from_sparse_dem(
        &dem,
        TrellisConfig {
            k: 8,
            delta: 100.0,
            score_alpha: 0.8,
            column_order: None,
            merge_indistinguishable: true,
            bp_score_iterations: 5,
            metric_mode: pecos_trellis::MetricMode::default(),
            int_metric_scale: 1024,
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
fn bptrellis_matches_hand_mapped_trellis_for_every_ordering() {
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
                escalation_ks: Vec::new(),
            },
        )
        .unwrap();
        let mut trellis = TrellisDecoder::from_sparse_dem(
            &dem,
            TrellisConfig {
                k: 4,
                delta: 12.0,
                score_alpha: 0.6,
                column_order,
                merge_indistinguishable: true,
                bp_score_iterations: 0,
                metric_mode: pecos_trellis::MetricMode::default(),
                int_metric_scale: 1024,
            },
        )
        .unwrap();

        for syndrome in [[0, 0], [1, 0], [0, 1], [1, 1]] {
            assert_eq!(
                decoder.decode(&syndrome).unwrap(),
                trellis.decode(&syndrome).unwrap()
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
fn bptrellis_observable_trait_rejects_wide_masks_without_truncation() {
    let dem = sparse_dem(vec![(0.2, vec![0], vec![64])], 1, 65);
    let decoder = BpTrellisDecoder::from_sparse_dem(&dem, BpTrellisConfig::default()).unwrap();
    let mut boxed: Box<dyn ObservableDecoder> = Box::new(decoder);

    assert!(boxed.decode_obs(&[1]).unwrap().get(64));
    assert!(matches!(
        boxed.decode_to_observables(&[1]),
        Err(DecoderError::InvalidConfiguration(_))
    ));
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

fn overpruning_escalation_dem() -> SparseDem {
    // The final mechanism is the only way to fire D2. Matching D0=D1=0 then
    // requires both less-likely prefixes, which K=2 has already discarded.
    sparse_dem(
        vec![
            (0.4, vec![0], vec![]),
            (0.4, vec![1], vec![]),
            (0.1, vec![0, 1, 2], vec![0]),
        ],
        3,
        1,
    )
}

fn escalation_config(k: usize, escalation_ks: Vec<usize>) -> BpTrellisConfig {
    BpTrellisConfig {
        k,
        delta: f64::INFINITY,
        score_alpha: 0.0,
        bp_score_iterations: 0,
        merge_indistinguishable: false,
        ordering: TrellisOrdering::TimeOrder,
        escalation_ks,
    }
}

#[test]
fn no_path_escalates_to_k16_and_accumulates_transitions() {
    let dem = overpruning_escalation_dem();
    let syndrome = [0, 0, 1];
    let mut base =
        BpTrellisDecoder::from_sparse_dem(&dem, escalation_config(2, Vec::new())).unwrap();
    assert!(base.decode(&syndrome).is_err());

    let mut bare_k16 =
        BpTrellisDecoder::from_sparse_dem(&dem, escalation_config(16, Vec::new())).unwrap();
    let bare_result = bare_k16.decode(&syndrome).unwrap();

    let mut ladder =
        BpTrellisDecoder::from_sparse_dem(&dem, escalation_config(2, vec![16])).unwrap();
    let result = ladder.decode(&syndrome).unwrap();

    assert_eq!(result.predicted, bare_result.predicted);
    assert_eq!(result.escalation_rungs_used, 1);
    assert!(result.transitions > bare_result.transitions);
}

#[test]
fn exhausted_ladder_propagates_the_final_rung_error() {
    let dem = overpruning_escalation_dem();
    let syndrome = [0, 0, 1];
    let mut bare_final =
        BpTrellisDecoder::from_sparse_dem(&dem, escalation_config(2, Vec::new())).unwrap();
    let expected = bare_final.decode(&syndrome).unwrap_err();

    let mut ladder =
        BpTrellisDecoder::from_sparse_dem(&dem, escalation_config(1, vec![2])).unwrap();
    let actual = ladder.decode(&syndrome).unwrap_err();

    assert!(matches!(actual, DecoderError::DecodingFailed(_)));
    assert_eq!(actual.to_string(), expected.to_string());
}

#[test]
fn successful_base_decode_is_bit_identical_with_a_configured_ladder() {
    let dem = sparse_dem(vec![(0.25, vec![], vec![0])], 0, 1);
    let mut bare =
        BpTrellisDecoder::from_sparse_dem(&dem, escalation_config(1, Vec::new())).unwrap();
    let mut ladder =
        BpTrellisDecoder::from_sparse_dem(&dem, escalation_config(1, vec![16])).unwrap();

    let bare_result = bare.decode(&[]).unwrap();
    let ladder_result = ladder.decode(&[]).unwrap();

    assert_eq!(ladder_result.escalation_rungs_used, 0);
    assert_eq!(ladder_result, bare_result);
}

#[test]
fn wrong_prediction_does_not_escalate() {
    let dem = sparse_dem(
        vec![
            (0.20, vec![0], vec![]),
            (0.20, vec![0], vec![]),
            (0.30, vec![0], vec![0]),
        ],
        1,
        1,
    );
    let mut exact =
        BpTrellisDecoder::from_sparse_dem(&dem, escalation_config(16, Vec::new())).unwrap();
    assert!(exact.decode(&[1]).unwrap().predicted.is_zero());

    let mut ladder =
        BpTrellisDecoder::from_sparse_dem(&dem, escalation_config(1, vec![16])).unwrap();
    let result = ladder.decode(&[1]).unwrap();

    assert_eq!(result.predicted, ObsMask::from_u64(1));
    assert_eq!(result.escalation_rungs_used, 0);
}

#[test]
fn ladder_rung_matches_a_hand_built_decoder_except_accumulated_work() {
    let dem = sparse_dem(
        vec![
            (0.20, vec![0], vec![]),
            (0.25, vec![0], vec![]),
            (0.40, vec![1], vec![]),
            (0.10, vec![0, 1, 2], vec![0]),
        ],
        3,
        1,
    );
    let config_for = |k, escalation_ks| BpTrellisConfig {
        k,
        delta: f64::INFINITY,
        score_alpha: 0.0,
        bp_score_iterations: 0,
        merge_indistinguishable: true,
        ordering: TrellisOrdering::Explicit(vec![1, 2, 3, 0]),
        escalation_ks,
    };
    let syndrome = [0, 0, 1];
    let mut ladder = BpTrellisDecoder::from_sparse_dem(&dem, config_for(2, vec![16])).unwrap();
    let mut hand_built =
        BpTrellisDecoder::from_sparse_dem(&dem, config_for(16, Vec::new())).unwrap();

    let mut actual = ladder.decode(&syndrome).unwrap();
    let expected = hand_built.decode(&syndrome).unwrap();

    assert_eq!(actual.escalation_rungs_used, 1);
    assert!(actual.transitions > expected.transitions);
    assert_eq!(
        actual.processed_columns, 3,
        "the rung must preserve merging"
    );
    actual.transitions = expected.transitions;
    actual.escalation_rungs_used = 0;
    assert_eq!(actual, expected);
}

#[test]
fn bptrellis_escalates_through_observable_decoder_trait_object() {
    let dem = overpruning_escalation_dem();
    let decoder = BpTrellisDecoder::from_sparse_dem(&dem, escalation_config(2, vec![16])).unwrap();
    let mut boxed: Box<dyn ObservableDecoder> = Box::new(decoder);

    assert_eq!(boxed.decode_to_observables(&[0, 0, 1]).unwrap(), 1);
}
