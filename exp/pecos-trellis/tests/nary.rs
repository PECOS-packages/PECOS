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
        let expected_winner = expected_log_masses
            .iter()
            .min_by(|(left_label, left_mass), (right_label, right_mass)| {
                right_mass
                    .total_cmp(left_mass)
                    .then_with(|| left_label.cmp(right_label))
            })
            .map(|(&logical, _)| logical)
            .unwrap();
        assert_eq!(words_to_u64(result.predicted.words()), expected_winner);
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
                ..exact_config()
            },
        ),
        "BP-guided pruning requires a binary model",
    );
    assert_decode_matches_enumeration(&model, usize::MAX);
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
    let model = FactorModel::from(&dem);
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
                ..exact_config()
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
    let model = FactorModel::from(&dem);
    assert_eq!(
        deadline_column_order_for_factors(&model).unwrap(),
        deadline_column_order(&dem).unwrap()
    );
}
