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

use pecos_trellis::factor::{Factor, FactorModel, Outcome};
use pecos_trellis::{
    DecoderError, SparseDem, TrellisConfig, TrellisDecoder, TrellisResult, TrellisStatus,
    deadline_column_order, deadline_column_order_for_factors,
};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::BTreeMap;

fn exact_config() -> TrellisConfig {
    TrellisConfig {
        k: usize::MAX,
        delta: f64::INFINITY,
        score_alpha: 0.8,
        column_order: None,
        merge_indistinguishable: false,
        bp_score_iterations: 0,
    }
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

fn outcome(probability: f64, detectors: &[u32], observables: &[u32]) -> Outcome {
    Outcome {
        probability,
        detectors: detectors.to_vec(),
        observables: observables.to_vec(),
    }
}

fn syndrome(mask: usize, num_detectors: usize) -> Vec<u8> {
    (0..num_detectors)
        .map(|detector| u8::from(mask & (1 << detector) != 0))
        .collect()
}

fn words_to_u64(words: &[u64]) -> u64 {
    assert!(words.iter().skip(1).all(|&word| word == 0));
    words.first().copied().unwrap_or(0)
}

fn actual_masses(result: &TrellisResult) -> BTreeMap<u64, f64> {
    result
        .logical_masses
        .iter()
        .map(|entry| (words_to_u64(entry.logical.words()), entry.log_mass))
        .collect()
}

fn assert_result_masses_bitwise_equal(left: &TrellisResult, right: &TrellisResult) {
    assert_eq!(left.predicted, right.predicted);
    assert_eq!(left.log_evidence.to_bits(), right.log_evidence.to_bits());
    assert_eq!(
        left.runner_up_gap.map(f64::to_bits),
        right.runner_up_gap.map(f64::to_bits)
    );
    assert_eq!(left.logical_masses.len(), right.logical_masses.len());
    for (left_mass, right_mass) in left.logical_masses.iter().zip(&right.logical_masses) {
        assert_eq!(left_mass.logical, right_mass.logical);
        assert_eq!(left_mass.log_mass.to_bits(), right_mass.log_mass.to_bits());
    }
}

fn enumerate_factor_model(model: &FactorModel) -> BTreeMap<(u64, u64), f64> {
    fn visit(
        model: &FactorModel,
        factor_index: usize,
        detector_mask: u64,
        logical_mask: u64,
        mass: f64,
        totals: &mut BTreeMap<(u64, u64), f64>,
    ) {
        if factor_index == model.factors().len() {
            *totals.entry((detector_mask, logical_mask)).or_default() += mass;
            return;
        }
        for outcome in &model.factors()[factor_index].outcomes {
            let outcome_detectors = outcome
                .detectors
                .iter()
                .fold(0, |mask, &detector| mask ^ (1 << detector));
            let outcome_observables = outcome
                .observables
                .iter()
                .fold(0, |mask, &observable| mask ^ (1 << observable));
            visit(
                model,
                factor_index + 1,
                detector_mask ^ outcome_detectors,
                logical_mask ^ outcome_observables,
                mass * outcome.probability,
                totals,
            );
        }
    }

    let mut totals = BTreeMap::new();
    visit(model, 0, 0, 0, 1.0, &mut totals);
    totals
}

fn assert_decode_matches_enumeration(model: &FactorModel, case_index: usize) {
    let enumerated = enumerate_factor_model(model);
    let mut decoder = TrellisDecoder::from_factor_model(model, exact_config()).unwrap();
    for syndrome_mask in 0..(1 << model.num_detectors()) {
        let expected: BTreeMap<u64, f64> = enumerated
            .iter()
            .filter_map(|(&(detectors, logical), &mass)| {
                (detectors == syndrome_mask as u64 && mass > 0.0).then_some((logical, mass))
            })
            .collect();
        let decoded = decoder.decode(&syndrome(syndrome_mask, model.num_detectors()));
        if expected.is_empty() {
            assert!(
                matches!(decoded, Err(DecoderError::DecodingFailed(_))),
                "case {case_index}, syndrome {syndrome_mask}: expected no path, got {decoded:?}"
            );
            continue;
        }

        let result = decoded.unwrap();
        assert_eq!(result.status, TrellisStatus::Exact);
        let expected_log_masses: BTreeMap<u64, f64> = expected
            .iter()
            .map(|(&logical, &mass)| (logical, mass.ln()))
            .collect();
        let actual = actual_masses(&result);
        assert_eq!(actual.len(), expected_log_masses.len());
        for (logical, expected_log_mass) in &expected_log_masses {
            let actual_log_mass = actual.get(logical).unwrap_or_else(|| {
                panic!("case {case_index}, syndrome {syndrome_mask}: missing label {logical}")
            });
            assert!(
                (actual_log_mass - expected_log_mass).abs() <= 1e-9,
                "case {case_index}, syndrome {syndrome_mask}, label {logical}: expected {expected_log_mass}, got {actual_log_mass}"
            );
        }
        let expected_evidence = expected.values().sum::<f64>().ln();
        assert!((result.log_evidence - expected_evidence).abs() <= 1e-9);
        let predicted = words_to_u64(result.predicted.words());
        let predicted_mass = expected
            .get(&predicted)
            .unwrap_or_else(|| panic!("case {case_index}: predicted label {predicted} is absent"));
        let mut ranked_expected: Vec<(u64, f64)> = expected
            .iter()
            .map(|(&logical, &mass)| (logical, mass))
            .collect();
        ranked_expected.sort_by(|(left_label, left_mass), (right_label, right_mass)| {
            right_mass
                .total_cmp(left_mass)
                .then_with(|| left_label.cmp(right_label))
        });
        let (expected_winner, maximum_mass) = ranked_expected[0];
        assert!(
            maximum_mass - predicted_mass <= 1e-9,
            "case {case_index}: predicted label {predicted} has expected mass {predicted_mass}, below maximum {maximum_mass}"
        );
        if ranked_expected
            .get(1)
            .is_none_or(|(_, runner_up_mass)| maximum_mass - runner_up_mass > 1e-6)
        {
            assert_eq!(predicted, expected_winner);
        }
    }
}

#[test]
fn seeded_small_nary_models_match_brute_force_for_every_syndrome() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4e41_5259_5f4f_5241);
    for case_index in 0..36 {
        let num_detectors = rng.random_range(1..=4);
        let num_observables = rng.random_range(0..=2);
        let factor_count = rng.random_range(1..=4);
        let mut factors = Vec::with_capacity(factor_count);
        for factor_index in 0..factor_count {
            let outcome_count = if factor_index == 0 {
                3
            } else {
                rng.random_range(1..=3)
            };
            let probabilities = match outcome_count {
                1 => vec![1.0],
                2 => vec![0.65, 0.35],
                3 => vec![0.5, 0.3, 0.2],
                _ => unreachable!(),
            };
            let outcomes = probabilities
                .into_iter()
                .enumerate()
                .map(|(outcome_index, probability)| {
                    let detector_count = rng.random_range(0..=num_detectors.min(2));
                    let mut detectors = Vec::with_capacity(detector_count);
                    while detectors.len() < detector_count {
                        let detector = u32::try_from(rng.random_range(0..num_detectors))
                            .expect("generated detector index is at most three");
                        if !detectors.contains(&detector) {
                            detectors.push(detector);
                        }
                    }
                    if factor_index == 0 && outcome_index > 0 && detectors.is_empty() {
                        detectors.push(
                            u32::try_from((outcome_index - 1) % num_detectors)
                                .expect("generated detector index is at most three"),
                        );
                    }
                    detectors.sort_unstable();
                    let observables = if num_observables > 0 && rng.random_bool(0.5) {
                        vec![
                            u32::try_from(rng.random_range(0..num_observables))
                                .expect("generated observable index is at most one"),
                        ]
                    } else {
                        Vec::new()
                    };
                    outcome(probability, &detectors, &observables)
                })
                .collect();
            factors.push(Factor { outcomes });
        }
        let model = FactorModel::new(factors, num_detectors, num_observables).unwrap();
        assert_decode_matches_enumeration(&model, case_index);
    }
}

#[test]
fn drifted_binary_baseline_uses_stored_probability_in_nary_kernel() {
    let p = 0.999_999_f64;
    let complement = 1.0 - p;
    let q = complement + 5e-11;
    assert_ne!(q.to_bits(), complement.to_bits());
    let model = FactorModel::new(
        vec![Factor {
            outcomes: vec![outcome(q, &[], &[]), outcome(p, &[0], &[0])],
        }],
        1,
        1,
    )
    .unwrap();

    assert_invalid(
        TrellisDecoder::from_factor_model(
            &model,
            TrellisConfig {
                bp_score_iterations: 1,
                ..TrellisConfig::default()
            },
        ),
        "BP-guided pruning requires a binary model",
    );
    assert_decode_matches_enumeration(&model, usize::MAX);
}

#[test]
fn decimal_literal_binary_pairs_delegate_in_both_probability_orders() {
    for (baseline_probability, toggle_probability, observable) in [(0.2, 0.8, 0), (0.8, 0.2, 1)] {
        let model = FactorModel::new(
            vec![Factor {
                outcomes: vec![
                    outcome(baseline_probability, &[], &[]),
                    outcome(toggle_probability, &[0], &[observable]),
                ],
            }],
            1,
            2,
        )
        .unwrap();
        let dem = sparse_dem(vec![(toggle_probability, vec![0], vec![observable])], 1, 2);
        let config = TrellisConfig {
            bp_score_iterations: 2,
            ..TrellisConfig::default()
        };
        let mut factor_decoder = TrellisDecoder::from_factor_model(&model, config.clone()).unwrap();
        let mut dem_decoder = TrellisDecoder::from_sparse_dem(&dem, config).unwrap();

        for observed in [[0], [1]] {
            let factor_result = factor_decoder.decode(&observed).unwrap();
            let dem_result = dem_decoder.decode(&observed).unwrap();
            assert_result_masses_bitwise_equal(&factor_result, &dem_result);
        }
    }
}

#[test]
fn one_in_a_million_baseline_delegates_in_both_listing_orders() {
    let baseline_probability = 1e-6_f64;
    let toggle_probability = 0.999_999_f64;
    assert_eq!(
        (baseline_probability + toggle_probability).to_bits(),
        1.0_f64.to_bits()
    );
    let dem = sparse_dem(vec![(toggle_probability, vec![0], vec![0])], 1, 1);

    for baseline_first in [true, false] {
        let baseline = outcome(baseline_probability, &[], &[]);
        let toggle = outcome(toggle_probability, &[0], &[0]);
        let outcomes = if baseline_first {
            vec![baseline, toggle]
        } else {
            vec![toggle, baseline]
        };
        let model = FactorModel::new(vec![Factor { outcomes }], 1, 1).unwrap();
        let config = TrellisConfig {
            bp_score_iterations: 2,
            ..TrellisConfig::default()
        };
        let mut factor_decoder = TrellisDecoder::from_factor_model(&model, config.clone()).unwrap();
        let mut dem_decoder = TrellisDecoder::from_sparse_dem(&dem, config).unwrap();

        for observed in [[0], [1]] {
            let factor_result = factor_decoder.decode(&observed).unwrap();
            let dem_result = dem_decoder.decode(&observed).unwrap();
            assert_result_masses_bitwise_equal(&factor_result, &dem_result);
        }
    }
}

#[test]
fn tiny_exact_sum_baseline_stays_nary_when_binary_encoding_exceeds_tolerance() {
    let baseline_probability = 1e-9_f64;
    let toggle_probability = 1.0 - baseline_probability;
    let model = FactorModel::new(
        vec![Factor {
            outcomes: vec![
                outcome(baseline_probability, &[], &[]),
                outcome(toggle_probability, &[0], &[0]),
            ],
        }],
        1,
        1,
    )
    .unwrap();

    // At this scale, ulp(1)-level subtraction noise changes the baseline log
    // mass by more than 1e-9. Refusing delegation is faithful evaluation, not
    // a missed binary optimization.
    assert_invalid(
        TrellisDecoder::from_factor_model(
            &model,
            TrellisConfig {
                bp_score_iterations: 2,
                ..TrellisConfig::default()
            },
        ),
        "BP-guided pruning requires a binary model",
    );
}

#[test]
fn seeded_complement_tolerance_sweep_classifies_exact_and_drifted_pairs() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x434f_4d50_4c45_4d45);
    let active_bp = TrellisConfig {
        bp_score_iterations: 2,
        ..TrellisConfig::default()
    };

    for exponent in 1..=7 {
        let baseline_probability = 10_f64.powi(-exponent);
        let toggle_probability = 1.0 - baseline_probability;
        for baseline_first in [true, false] {
            let observable = u32::from(rng.random_bool(0.5));
            let baseline = outcome(baseline_probability, &[], &[]);
            let toggle = outcome(toggle_probability, &[0], &[observable]);
            let outcomes = if baseline_first {
                vec![baseline, toggle]
            } else {
                vec![toggle, baseline]
            };
            let model = FactorModel::new(vec![Factor { outcomes }], 1, 2).unwrap();
            TrellisDecoder::from_factor_model(&model, active_bp.clone()).unwrap();
        }
    }

    for (baseline_probability, toggle_probability) in [(0.05, 0.95), (0.2, 0.8)] {
        for baseline_first in [true, false] {
            let baseline = outcome(baseline_probability, &[], &[]);
            let toggle = outcome(toggle_probability, &[0], &[0]);
            let outcomes = if baseline_first {
                vec![baseline, toggle]
            } else {
                vec![toggle, baseline]
            };
            let model = FactorModel::new(vec![Factor { outcomes }], 1, 1).unwrap();
            TrellisDecoder::from_factor_model(&model, active_bp.clone()).unwrap();
        }
    }

    for exponent in 2..=7 {
        let complement = 10_f64.powi(-exponent);
        let baseline_probability = complement + 5e-11;
        let toggle_probability = 1.0 - complement;
        for baseline_first in [true, false] {
            let baseline = outcome(baseline_probability, &[], &[]);
            let toggle = outcome(toggle_probability, &[0], &[0]);
            let outcomes = if baseline_first {
                vec![baseline, toggle]
            } else {
                vec![toggle, baseline]
            };
            let model = FactorModel::new(vec![Factor { outcomes }], 1, 1).unwrap();
            assert_invalid(
                TrellisDecoder::from_factor_model(&model, active_bp.clone()),
                "BP-guided pruning requires a binary model",
            );
        }
    }
}

#[test]
fn bp_iterations_are_inert_for_unpruned_nary_decode() {
    let model = genuine_model();
    let mut without_bp = TrellisDecoder::from_factor_model(&model, exact_config()).unwrap();
    let mut with_inert_bp = TrellisDecoder::from_factor_model(
        &model,
        TrellisConfig {
            bp_score_iterations: 3,
            ..exact_config()
        },
    )
    .unwrap();

    for observed in [[0], [1]] {
        let without_bp_result = without_bp.decode(&observed).unwrap();
        let with_bp_result = with_inert_bp.decode(&observed).unwrap();
        assert_result_masses_bitwise_equal(&without_bp_result, &with_bp_result);
    }
}

fn symmetric_difference(left: &[u32], right: &[u32]) -> Vec<u32> {
    let mut parity = BTreeMap::new();
    for &index in left.iter().chain(right) {
        *parity.entry(index).or_insert(false) ^= true;
    }
    parity
        .into_iter()
        .filter_map(|(index, odd)| odd.then_some(index))
        .collect()
}

#[test]
fn nonempty_two_outcome_factors_match_equivalent_binary_dem() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4e41_5259_5f42_494e);
    for case_index in 0..20 {
        let num_detectors = rng.random_range(2..=5);
        let num_observables = 2;
        let factor_count = rng.random_range(1..=4);
        let mut factors = Vec::new();
        let mut mechanisms = Vec::new();
        for _ in 0..factor_count {
            let p = rng.random_range(0.05..0.45);
            let q = 1.0 - p;
            let a_detectors = vec![
                u32::try_from(rng.random_range(0..num_detectors))
                    .expect("generated detector index is at most four"),
            ];
            let b_detectors = vec![
                u32::try_from(rng.random_range(0..num_detectors))
                    .expect("generated detector index is at most four"),
            ];
            let a_observables = vec![
                u32::try_from(rng.random_range(0..num_observables))
                    .expect("generated observable index is at most one"),
            ];
            let b_observables = vec![
                u32::try_from(rng.random_range(0..num_observables))
                    .expect("generated observable index is at most one"),
            ];
            factors.push(Factor {
                outcomes: vec![
                    outcome(q, &a_detectors, &a_observables),
                    outcome(p, &b_detectors, &b_observables),
                ],
            });
            mechanisms.push((1.0, a_detectors.clone(), a_observables.clone()));
            mechanisms.push((
                p,
                symmetric_difference(&a_detectors, &b_detectors),
                symmetric_difference(&a_observables, &b_observables),
            ));
        }
        let model = FactorModel::new(factors, num_detectors, num_observables).unwrap();
        let dem = sparse_dem(mechanisms, num_detectors, num_observables);
        let mut nary = TrellisDecoder::from_factor_model(&model, exact_config()).unwrap();
        let mut binary = TrellisDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
        for syndrome_mask in 0..(1 << num_detectors) {
            let observed = syndrome(syndrome_mask, num_detectors);
            let nary_result = nary.decode(&observed);
            let binary_result = binary.decode(&observed);
            match (nary_result, binary_result) {
                (Ok(nary_result), Ok(binary_result)) => {
                    assert_eq!(nary_result.predicted, binary_result.predicted);
                    assert!((nary_result.log_evidence - binary_result.log_evidence).abs() <= 1e-9);
                    let nary_masses = actual_masses(&nary_result);
                    let binary_masses = actual_masses(&binary_result);
                    assert_eq!(nary_masses.len(), binary_masses.len());
                    for (logical, binary_mass) in binary_masses {
                        assert!(
                            (nary_masses[&logical] - binary_mass).abs() <= 1e-9,
                            "case {case_index}, syndrome {syndrome_mask}, label {logical}"
                        );
                    }
                }
                (Err(DecoderError::DecodingFailed(_)), Err(DecoderError::DecodingFailed(_))) => {}
                (nary_result, binary_result) => panic!(
                    "case {case_index}, syndrome {syndrome_mask}: N-ary {nary_result:?}, binary {binary_result:?}"
                ),
            }
        }
    }
}

#[test]
fn sparse_dem_factor_conversion_delegates_bitwise_with_binary_features() {
    let dem = sparse_dem(
        vec![
            (0.12, vec![0], vec![0]),
            (0.21, vec![0, 1], vec![1]),
            (0.21, vec![0, 1], vec![1]),
            (0.08, vec![1, 2], vec![0]),
            (1.0, vec![2], vec![1]),
        ],
        3,
        2,
    );
    let model = FactorModel::try_from(&dem).unwrap();
    let config = TrellisConfig {
        k: 3,
        delta: 20.0,
        score_alpha: 0.8,
        column_order: Some(vec![4, 2, 0, 3, 1]),
        merge_indistinguishable: true,
        bp_score_iterations: 4,
    };
    let mut direct = TrellisDecoder::from_sparse_dem(&dem, config.clone()).unwrap();
    let mut delegated = TrellisDecoder::from_factor_model(&model, config).unwrap();
    for syndrome_mask in 0..8 {
        let observed = syndrome(syndrome_mask, 3);
        match (direct.decode(&observed), delegated.decode(&observed)) {
            (Ok(direct), Ok(delegated)) => {
                assert_eq!(direct.predicted, delegated.predicted);
                assert_eq!(
                    direct.log_evidence.to_bits(),
                    delegated.log_evidence.to_bits()
                );
                assert_eq!(
                    direct.runner_up_gap.map(f64::to_bits),
                    delegated.runner_up_gap.map(f64::to_bits)
                );
                assert_eq!(direct.logical_masses.len(), delegated.logical_masses.len());
                for (direct, delegated) in
                    direct.logical_masses.iter().zip(&delegated.logical_masses)
                {
                    assert_eq!(direct.logical, delegated.logical);
                    assert_eq!(direct.log_mass.to_bits(), delegated.log_mass.to_bits());
                }
            }
            (Err(DecoderError::DecodingFailed(_)), Err(DecoderError::DecodingFailed(_))) => {}
            (direct, delegated) => panic!("direct {direct:?}, delegated {delegated:?}"),
        }
    }
}

fn genuine_model() -> FactorModel {
    FactorModel::new(
        vec![Factor {
            outcomes: vec![
                outcome(0.5, &[], &[]),
                outcome(0.3, &[0], &[]),
                outcome(0.2, &[0], &[0]),
            ],
        }],
        1,
        1,
    )
    .unwrap()
}

fn assert_invalid<T: std::fmt::Debug>(result: Result<T, DecoderError>, text: &str) {
    let error = result.expect_err("configuration must be rejected");
    assert!(
        matches!(error, DecoderError::InvalidConfiguration(_)),
        "wrong error kind: {error}"
    );
    assert!(
        error.to_string().contains(text),
        "unexpected error: {error}"
    );
}

#[test]
fn factor_model_validation_and_nary_feature_guards_fail_loud() {
    assert_invalid(
        FactorModel::new(
            vec![Factor {
                outcomes: vec![outcome(0.4, &[], &[]), outcome(0.5, &[0], &[])],
            }],
            1,
            0,
        ),
        "must sum to 1",
    );
    assert_invalid(
        FactorModel::new(vec![Factor { outcomes: vec![] }], 1, 1),
        "at least one outcome",
    );
    assert_invalid(
        FactorModel::new(
            vec![Factor {
                outcomes: vec![outcome(1.0, &[1], &[])],
            }],
            1,
            0,
        ),
        "detector index 1 is out of range",
    );
    assert_invalid(
        FactorModel::new(
            vec![Factor {
                outcomes: vec![outcome(1.0, &[], &[1])],
            }],
            0,
            1,
        ),
        "observable index 1 is out of range",
    );
    assert_invalid(
        FactorModel::new(
            vec![Factor {
                outcomes: vec![outcome(1.0, &[0, 0], &[])],
            }],
            1,
            0,
        ),
        "repeats detector index 0",
    );

    let model = genuine_model();
    assert_invalid(
        TrellisDecoder::from_factor_model(
            &model,
            TrellisConfig {
                bp_score_iterations: 1,
                ..TrellisConfig::default()
            },
        ),
        "BP-guided pruning requires a binary model",
    );
    assert_invalid(
        TrellisDecoder::from_factor_model(
            &model,
            TrellisConfig {
                merge_indistinguishable: true,
                ..exact_config()
            },
        ),
        "defined for binary mechanisms only",
    );
    assert_invalid(
        TrellisDecoder::from_factor_model(
            &model,
            TrellisConfig {
                column_order: Some(vec![]),
                ..exact_config()
            },
        ),
        "column_order must be a permutation",
    );

    let two_factors = FactorModel::new(
        vec![model.factors()[0].clone(), model.factors()[0].clone()],
        1,
        1,
    )
    .unwrap();
    assert_invalid(
        TrellisDecoder::from_factor_model(
            &two_factors,
            TrellisConfig {
                column_order: Some(vec![0, 0]),
                ..exact_config()
            },
        ),
        "column_order must be a permutation",
    );
}

#[test]
fn factor_deadline_order_matches_binary_image() {
    let dem = sparse_dem(
        vec![
            (0.2, vec![0, 2], vec![0]),
            (0.3, vec![1], vec![]),
            (0.1, vec![2, 3], vec![0]),
            (0.4, vec![], vec![0]),
        ],
        4,
        1,
    );
    let model = FactorModel::try_from(&dem).unwrap();
    assert_eq!(
        deadline_column_order_for_factors(&model).unwrap(),
        deadline_column_order(&dem).unwrap()
    );
}

#[test]
fn sparse_dem_conversion_rejects_duplicate_indices_before_factor_ordering() {
    let dem = sparse_dem(vec![(0.2, vec![0, 0], vec![])], 1, 0);
    let binary_error = deadline_column_order(&dem).unwrap_err();
    assert!(
        binary_error
            .to_string()
            .contains("mechanism 0 repeats detector index 0")
    );

    let factor_error = FactorModel::try_from(&dem).unwrap_err();
    assert!(
        factor_error
            .to_string()
            .contains("factor 0 outcome 1 repeats detector index 0")
    );
}

#[test]
fn both_empty_two_outcome_factors_classify_and_decode_order_independently() {
    // The degenerate pair toggles nothing, so listing order must not change
    // classification or a single bit of the decode. The drifted pair below
    // was the executed counterexample against position-based toggle choice:
    // p_a = 1e-3, p_b = 0.999 - 5e-12 (sum residual -5e-12, valid).
    let p_a = 1e-3_f64;
    let p_b = 0.999_f64 - 5e-12;
    for (first, second) in [(p_a, p_b), (p_b, p_a)] {
        let build = |x: f64, y: f64| {
            FactorModel::new(
                vec![
                    Factor {
                        outcomes: vec![outcome(x, &[], &[]), outcome(y, &[], &[])],
                    },
                    Factor {
                        outcomes: vec![outcome(0.8, &[], &[]), outcome(0.2, &[0], &[0])],
                    },
                ],
                1,
                1,
            )
            .unwrap()
        };
        let mut forward =
            TrellisDecoder::from_factor_model(&build(first, second), exact_config()).unwrap();
        let mut reversed =
            TrellisDecoder::from_factor_model(&build(second, first), exact_config()).unwrap();
        for syndrome_mask in 0..2_usize {
            let observed = syndrome(syndrome_mask, 1);
            let forward_result = forward.decode(&observed).unwrap();
            let reversed_result = reversed.decode(&observed).unwrap();
            assert_result_masses_bitwise_equal(&forward_result, &reversed_result);
        }
    }
}
