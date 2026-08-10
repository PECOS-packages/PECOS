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

use pecos_decoder_core::ObservableDecoder;
use pecos_decoder_core::dem::SparseDem;
use pecos_decoder_core::obs_mask::ObsMask;
use pecos_frontier::{
    CommitteeDirection, CommitteeStatus, FrontierCommittee, FrontierConfig,
    backward_deadline_column_order, deadline_column_order,
};
use serde::Deserialize;
use std::collections::BTreeMap;

const FIXTURES_JSON: &str = include_str!("fixtures/upstream_order_fixtures.json");

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct FixtureFile {
    generator: String,
    fixtures: Vec<Fixture>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Fixture {
    name: String,
    mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    num_detectors: usize,
    num_observables: usize,
    syndromes: Vec<u128>,
    pruned: PruningConfig,
    forward_ordering: Vec<usize>,
    backward_ordering: Vec<usize>,
    expected_committee: Vec<ExpectedCommitteeResult>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PruningConfig {
    k: usize,
    delta: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectedCommitteeResult {
    syndrome: u128,
    status: String,
    logical_hat: Option<u128>,
    direction: String,
    log_evidence: Option<f64>,
    engine: String,
}

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

fn dense_syndrome(mask: u128, num_detectors: usize) -> Vec<u8> {
    (0..num_detectors)
        .map(|bit| u8::from(mask & (1_u128 << bit) != 0))
        .collect()
}

fn mask_as_u128(mask: &ObsMask) -> u128 {
    assert!(
        mask.words().iter().skip(2).all(|&word| word == 0),
        "fixture labels must fit in u128"
    );
    u128::from(mask.words().first().copied().unwrap_or(0))
        | (u128::from(mask.words().get(1).copied().unwrap_or(0)) << 64)
}

fn expected_direction(direction: &str) -> CommitteeDirection {
    match direction {
        "forward" => CommitteeDirection::Forward,
        "backward" => CommitteeDirection::Backward,
        unknown => panic!("unknown committee direction {unknown}"),
    }
}

#[test]
fn ordering_and_committee_match_upstream_fixtures() {
    let fixture_file: FixtureFile =
        serde_json::from_str(FIXTURES_JSON).expect("order fixtures must parse");
    assert_eq!(
        fixture_file.generator,
        "generate_upstream_order_fixtures.py"
    );
    let mut backward_selections = 0;
    let mut single_leg_selections = 0;

    for fixture in fixture_file.fixtures {
        let dem = sparse_dem(
            fixture.mechanisms,
            fixture.num_detectors,
            fixture.num_observables,
        );
        assert_eq!(
            deadline_column_order(&dem).unwrap(),
            fixture.forward_ordering,
            "{}: forward ordering",
            fixture.name
        );
        assert_eq!(
            backward_deadline_column_order(&dem).unwrap(),
            fixture.backward_ordering,
            "{}: backward ordering",
            fixture.name
        );

        // The fixture mechanisms are in time order. Passing forward_ordering
        // builds the same forward-ordered model used by the upstream generator;
        // FrontierCommittee then makes its second leg by plain reversal.
        let mut committee = FrontierCommittee::from_sparse_dem(
            &dem,
            FrontierConfig {
                k: fixture.pruned.k,
                delta: fixture.pruned.delta,
                score_alpha: 0.8,
                label_diverse_retention: false,
                column_order: Some(fixture.forward_ordering.clone()),
                merge_indistinguishable: false,
                bp_score_iterations: 0,
            },
        )
        .unwrap();

        let mut expected_by_syndrome = BTreeMap::new();
        for expected in fixture.expected_committee {
            assert!(
                expected_by_syndrome
                    .insert(expected.syndrome, expected)
                    .is_none(),
                "{}: duplicate expected syndrome",
                fixture.name
            );
        }
        assert_eq!(
            expected_by_syndrome.len(),
            fixture.syndromes.len(),
            "{}: committee result count",
            fixture.name
        );

        for syndrome_mask in fixture.syndromes {
            let expected = expected_by_syndrome
                .remove(&syndrome_mask)
                .unwrap_or_else(|| panic!("{}: missing expected syndrome", fixture.name));
            assert_eq!(expected.engine, "native_binary");
            let syndrome = dense_syndrome(syndrome_mask, fixture.num_detectors);
            let decoded = committee.decode(&syndrome);

            match expected.status.as_str() {
                "ok" => {
                    let result = decoded.unwrap_or_else(|error| {
                        panic!(
                            "{} syndrome {syndrome_mask}: expected success, got {error}",
                            fixture.name
                        )
                    });
                    let direction = expected_direction(&expected.direction);
                    assert_eq!(
                        result.direction, direction,
                        "{} syndrome {syndrome_mask}: direction",
                        fixture.name
                    );
                    assert_eq!(
                        mask_as_u128(&result.selected.predicted),
                        expected.logical_hat.expect("ok result needs logical_hat"),
                        "{} syndrome {syndrome_mask}: logical label",
                        fixture.name
                    );
                    let expected_evidence =
                        expected.log_evidence.expect("ok result needs log_evidence");
                    assert!(
                        (result.selected.log_evidence - expected_evidence).abs() <= 1e-9,
                        "{} syndrome {syndrome_mask}: expected evidence {expected_evidence}, got {}",
                        fixture.name,
                        result.selected.log_evidence
                    );
                    let selected_member = match result.direction {
                        CommitteeDirection::Forward => result.forward,
                        CommitteeDirection::Backward => result.backward,
                    };
                    assert_eq!(selected_member.status, CommitteeStatus::Ok);
                    assert_eq!(
                        selected_member.log_evidence.to_bits(),
                        result.selected.log_evidence.to_bits()
                    );
                    if result.direction == CommitteeDirection::Backward {
                        backward_selections += 1;
                    }
                    if result.forward.status != result.backward.status {
                        single_leg_selections += 1;
                        let failed_member = if result.forward.status == CommitteeStatus::NoPath {
                            result.forward
                        } else {
                            result.backward
                        };
                        assert_eq!(
                            failed_member.log_evidence.to_bits(),
                            f64::NEG_INFINITY.to_bits()
                        );
                    }
                }
                "no_path" => {
                    assert!(expected.logical_hat.is_none());
                    assert!(expected.log_evidence.is_none());
                    assert!(
                        decoded.is_err(),
                        "{} syndrome {syndrome_mask}",
                        fixture.name
                    );
                }
                status => panic!("{}: unknown committee status {status}", fixture.name),
            }
        }
        assert!(expected_by_syndrome.is_empty());
    }

    assert!(backward_selections > 0, "fixtures must select backward");
    assert!(
        single_leg_selections > 0,
        "fixtures must exercise one-leg no-path selection"
    );
}

#[test]
fn deadline_order_closes_earlier_rows_first() {
    // D0 has first/last touches 0/2; D1 has 1/3. The keys therefore place
    // both D0 columns before both D1 columns, preserving original-index ties:
    // [0(D0), 2(D0), 1(D1), 3(D1)].
    let dem = sparse_dem(
        vec![
            (0.1, vec![0], vec![]),
            (0.1, vec![1], vec![]),
            (0.1, vec![0], vec![]),
            (0.1, vec![1], vec![]),
        ],
        2,
        0,
    );
    assert_eq!(deadline_column_order(&dem).unwrap(), vec![0, 2, 1, 3]);
}

#[test]
fn detector_free_mechanisms_sort_last_and_empty_dem_stays_empty() {
    let dem = sparse_dem(vec![(0.1, vec![], vec![0]), (0.1, vec![0], vec![])], 1, 1);
    assert_eq!(deadline_column_order(&dem).unwrap(), vec![1, 0]);

    let empty = sparse_dem(Vec::new(), 0, 0);
    assert!(deadline_column_order(&empty).unwrap().is_empty());
    assert!(backward_deadline_column_order(&empty).unwrap().is_empty());
}

#[test]
fn ordering_rejects_invalid_and_duplicate_detector_indices() {
    let out_of_range = sparse_dem(vec![(0.1, vec![1], vec![])], 1, 0);
    assert!(deadline_column_order(&out_of_range).is_err());

    let duplicate = sparse_dem(vec![(0.1, vec![0, 0], vec![])], 1, 0);
    assert!(backward_deadline_column_order(&duplicate).is_err());
}

#[test]
fn committee_ties_select_forward_and_trait_uses_selected_mask() {
    let mut committee = FrontierCommittee::from_dem_str("", FrontierConfig::default()).unwrap();
    let result = committee.decode(&[]).unwrap();
    assert_eq!(result.direction, CommitteeDirection::Forward);
    assert_eq!(result.forward.status, CommitteeStatus::Ok);
    assert_eq!(result.backward.status, CommitteeStatus::Ok);

    let mut boxed: Box<dyn ObservableDecoder> = Box::new(committee);
    assert!(boxed.decode_obs(&[]).unwrap().is_zero());
}
