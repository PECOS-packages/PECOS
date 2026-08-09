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

use pecos_decoder_core::dem::SparseDem;
use pecos_decoder_core::obs_mask::ObsMask;
use pecos_decoder_core::{DecoderError, ObservableDecoder};
use pecos_frontier::{FrontierConfig, FrontierDecoder, FrontierResult, FrontierStatus};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::{BTreeMap, BTreeSet};

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

fn exact_config() -> FrontierConfig {
    FrontierConfig {
        k: usize::MAX,
        delta: f64::INFINITY,
        score_alpha: 0.8,
        column_order: None,
        merge_indistinguishable: false,
        bp_score_iterations: 0,
    }
}

fn merged_exact_config() -> FrontierConfig {
    FrontierConfig {
        merge_indistinguishable: true,
        ..exact_config()
    }
}

fn independent_logaddexp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let high = left.max(right);
    let low = left.min(right);
    high + (low - high).exp().ln_1p()
}

fn independent_enumeration(dem: &SparseDem, observed: &[u8]) -> BTreeMap<Vec<u64>, f64> {
    let logical_words = dem.num_observables.div_ceil(u64::BITS as usize);
    let mut masses = BTreeMap::new();
    for subset in 0..(1usize << dem.mechanisms.len()) {
        let mut detectors = vec![0_u8; dem.num_detectors];
        let mut logical = vec![0_u64; logical_words];
        let mut log_mass = 0.0;
        for (index, (probability, detector_set, observable_set)) in
            dem.mechanisms.iter().enumerate()
        {
            if subset & (1 << index) == 0 {
                log_mass += (1.0 - probability).ln();
            } else {
                log_mass += probability.ln();
                for &detector in detector_set {
                    detectors[detector as usize] ^= 1;
                }
                for &observable in observable_set {
                    logical[observable as usize / 64] ^= 1 << (observable % 64);
                }
            }
        }
        if log_mass.is_finite() && detectors == observed {
            masses
                .entry(logical)
                .and_modify(|mass| *mass = independent_logaddexp(*mass, log_mass))
                .or_insert(log_mass);
        }
    }
    masses
}

fn numeric_words_cmp(left: &[u64], right: &[u64]) -> std::cmp::Ordering {
    left.iter().rev().cmp(right.iter().rev())
}

fn independent_winner(masses: &BTreeMap<Vec<u64>, f64>) -> Vec<u64> {
    masses
        .iter()
        .min_by(|(left_label, left_mass), (right_label, right_mass)| {
            right_mass
                .total_cmp(left_mass)
                .then_with(|| numeric_words_cmp(left_label, right_label))
        })
        .map(|(label, _)| label.clone())
        .expect("generated syndrome must have at least one explanation")
}

fn result_mass_map(result: &FrontierResult) -> BTreeMap<Vec<u64>, f64> {
    result
        .logical_masses
        .iter()
        .map(|entry| (entry.logical.words().to_vec(), entry.log_mass))
        .collect()
}

fn assert_result_semantics_bitwise_equal(left: &FrontierResult, right: &FrontierResult) {
    assert_eq!(left.predicted, right.predicted);
    assert_eq!(left.log_evidence.to_bits(), right.log_evidence.to_bits());
    assert_eq!(
        left.runner_up_gap.map(f64::to_bits),
        right.runner_up_gap.map(f64::to_bits)
    );
    assert_eq!(left.peak_retained_states, right.peak_retained_states);
    assert_eq!(left.processed_columns, right.processed_columns);
    assert_eq!(left.transitions, right.transitions);
    assert_eq!(left.dropped_states, right.dropped_states);
    assert_eq!(
        left.dropped_log_mass.to_bits(),
        right.dropped_log_mass.to_bits()
    );
    assert_eq!(left.status, right.status);
    assert_eq!(left.logical_masses.len(), right.logical_masses.len());
    for (left_mass, right_mass) in left.logical_masses.iter().zip(&right.logical_masses) {
        assert_eq!(left_mass.logical, right_mass.logical);
        assert_eq!(left_mass.log_mass.to_bits(), right_mass.log_mass.to_bits());
    }
}

fn assert_mass_maps_close(
    actual: &BTreeMap<Vec<u64>, f64>,
    expected: &BTreeMap<Vec<u64>, f64>,
    tolerance: f64,
    context: &str,
) {
    assert_eq!(actual.len(), expected.len(), "{context}: label count");
    for (label, expected_mass) in expected {
        let actual_mass = actual
            .get(label)
            .unwrap_or_else(|| panic!("{context}: missing logical label {label:?}"));
        assert!(
            (actual_mass - expected_mass).abs() <= tolerance,
            "{context}, label {label:?}: expected {expected_mass}, got {actual_mass}"
        );
    }
}

#[test]
fn unpruned_matches_independent_brute_force_on_seeded_random_dems() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4652_4f4e_5449_4552);
    let mut forced_mechanism_count = 0;

    for case_index in 0..33 {
        let mechanism_count = 4 + case_index % 11;
        let num_detectors = rng.random_range(1..=10);
        let num_observables = rng.random_range(1..=3);
        let mut mechanisms = Vec::with_capacity(mechanism_count);
        let mut sampled_subset = Vec::with_capacity(mechanism_count);
        let mut observed = vec![0_u8; num_detectors];

        for mechanism_index in 0..mechanism_count {
            let forced = rng.random_bool(0.08);
            let probability = if forced {
                forced_mechanism_count += 1;
                1.0
            } else {
                rng.random_range(0.01..0.4)
            };
            let max_weight = num_detectors.min(3);
            let detector_count = if mechanism_index % 3 == 0 && max_weight >= 2 {
                rng.random_range(2..=max_weight)
            } else {
                1
            };
            let mut detectors = Vec::with_capacity(detector_count);
            while detectors.len() < detector_count {
                let detector = u32::try_from(rng.random_range(0..num_detectors))
                    .expect("random detector index is at most 9");
                if !detectors.contains(&detector) {
                    detectors.push(detector);
                }
            }
            detectors.sort_unstable();

            let observable_count = rng.random_range(0..=num_observables.min(2));
            let mut observables = Vec::with_capacity(observable_count);
            while observables.len() < observable_count {
                let observable = u32::try_from(rng.random_range(0..num_observables))
                    .expect("random observable index is at most 2");
                if !observables.contains(&observable) {
                    observables.push(observable);
                }
            }
            observables.sort_unstable();

            let taken = forced || rng.random_bool(0.35);
            if taken {
                for &detector in &detectors {
                    observed[detector as usize] ^= 1;
                }
            }
            sampled_subset.push(taken);
            mechanisms.push((probability, detectors, observables));
        }

        let dem = sparse_dem(mechanisms, num_detectors, num_observables);
        let mut config = exact_config();
        if case_index % 2 == 1 {
            config.column_order = Some((0..mechanism_count).rev().collect());
        }
        let mut decoder = FrontierDecoder::from_sparse_dem(&dem, config).unwrap();
        let result = decoder.decode(&observed).unwrap();
        let expected = independent_enumeration(&dem, &observed);
        let actual = result_mass_map(&result);

        assert_eq!(actual.len(), expected.len(), "case {case_index}");
        for (label, expected_mass) in &expected {
            let actual_mass = actual.get(label).expect("logical label must be retained");
            assert!(
                (actual_mass - expected_mass).abs() <= 1e-9,
                "case {case_index}, label {label:?}: expected {expected_mass}, got {actual_mass}"
            );
        }
        assert_eq!(
            result.predicted.words(),
            independent_winner(&expected),
            "case {case_index}, sampled subset {sampled_subset:?}"
        );
    }

    assert!(
        forced_mechanism_count > 0,
        "seeded models must exercise forced mechanisms"
    );
}

#[test]
fn indistinguishable_merging_matches_unmerged_and_original_brute_force() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4d30_425f_584f_525f);

    for case_index in 0..24 {
        let num_detectors = rng.random_range(3..=7);
        let num_observables = rng.random_range(1..=3);
        let base_count = 3 + case_index % 3;
        let duplicate_probabilities: Vec<f64> = if case_index % 2 == 0 {
            match case_index % 6 {
                0 => vec![0.3, 0.3],
                2 => vec![0.12, 0.47],
                _ => vec![0.5, 0.21],
            }
        } else {
            match case_index % 6 {
                1 => vec![0.5, 0.5, 0.5],
                3 => vec![0.1, 0.2, 0.3],
                _ => vec![0.17, 0.41, 0.26],
            }
        };
        let duplicate_observables = if case_index % 3 == 0 {
            Vec::new()
        } else {
            vec![0]
        };

        let mut base_mechanisms = Vec::with_capacity(base_count);
        for base_index in 0..base_count {
            let detector_count = 1 + usize::from(base_index % 3 == 0);
            let mut detectors = Vec::with_capacity(detector_count);
            while detectors.len() < detector_count {
                let detector = u32::try_from(rng.random_range(1..num_detectors))
                    .expect("seeded detector index fits u32");
                if !detectors.contains(&detector) {
                    detectors.push(detector);
                }
            }
            detectors.sort_unstable();
            let observables = if rng.random_bool(0.5) {
                vec![
                    u32::try_from(rng.random_range(0..num_observables))
                        .expect("seeded observable index fits u32"),
                ]
            } else {
                Vec::new()
            };
            base_mechanisms.push((rng.random_range(0.02..0.48), detectors, observables));
        }

        // Plant the first two equal-symptom copies around an unrelated column;
        // odd cases add a third copy later in the sequence.
        let mut mechanisms = Vec::with_capacity(base_count + duplicate_probabilities.len());
        mechanisms.push((
            duplicate_probabilities[0],
            vec![0],
            duplicate_observables.clone(),
        ));
        mechanisms.push(base_mechanisms.remove(0));
        mechanisms.push((
            duplicate_probabilities[1],
            vec![0],
            duplicate_observables.clone(),
        ));
        mechanisms.append(&mut base_mechanisms);
        if let Some(&third_probability) = duplicate_probabilities.get(2) {
            mechanisms.push((third_probability, vec![0], duplicate_observables));
        }

        let mut observed = vec![0_u8; num_detectors];
        for (probability, detectors, _) in &mechanisms {
            if rng.random_bool(*probability) {
                for &detector in detectors {
                    observed[detector as usize] ^= 1;
                }
            }
        }

        let dem = sparse_dem(mechanisms, num_detectors, num_observables);
        let mechanism_count = dem.mechanisms.len();
        let expected_merged_count = dem
            .mechanisms
            .iter()
            .map(|(_, detectors, observables)| (detectors, observables))
            .collect::<BTreeSet<_>>()
            .len();
        let order = (case_index % 2 == 1).then(|| (0..mechanism_count).rev().collect());
        let mut off_config = exact_config();
        off_config.column_order = order.clone();
        let mut on_config = merged_exact_config();
        on_config.column_order = order;
        let mut unmerged = FrontierDecoder::from_sparse_dem(&dem, off_config).unwrap();
        let mut merged = FrontierDecoder::from_sparse_dem(&dem, on_config).unwrap();
        let unmerged_result = unmerged.decode(&observed).unwrap();
        let merged_result = merged.decode(&observed).unwrap();
        let enumerated = independent_enumeration(&dem, &observed);
        let context = format!("seeded merge case {case_index}");

        assert_eq!(
            merged_result.predicted, unmerged_result.predicted,
            "{context}"
        );
        assert_eq!(
            merged_result.predicted.words(),
            independent_winner(&enumerated),
            "{context}"
        );
        assert_mass_maps_close(
            &result_mass_map(&merged_result),
            &result_mass_map(&unmerged_result),
            1e-12,
            &context,
        );
        assert_mass_maps_close(
            &result_mass_map(&merged_result),
            &enumerated,
            1e-12,
            &context,
        );
        assert_eq!(unmerged_result.processed_columns, mechanism_count);
        assert_eq!(merged_result.processed_columns, expected_merged_count);
    }
}

#[test]
fn merging_keeps_first_ordered_occurrence_and_deletes_later_copy() {
    // The original-index permutation produces [A(first), B, A(second), C].
    // Merging must produce [A(xor), B, C], not move A to its later slot.
    let dem = sparse_dem(
        vec![
            (0.2, vec![], vec![1]),     // C: original index 0
            (0.25, vec![0], vec![]),    // A(second): original index 1
            (0.1, vec![0, 1], vec![0]), // B: original index 2
            (0.2, vec![0], vec![]),     // A(first): original index 3
        ],
        2,
        2,
    );
    let xor_probability = 0.2 * (1.0 - 0.25) + 0.25 * (1.0 - 0.2);
    let premerged = sparse_dem(
        vec![
            (xor_probability, vec![0], vec![]),
            (0.1, vec![0, 1], vec![0]),
            (0.2, vec![], vec![1]),
        ],
        2,
        2,
    );
    let moved_to_last = sparse_dem(
        vec![
            (0.1, vec![0, 1], vec![0]),
            (xor_probability, vec![0], vec![]),
            (0.2, vec![], vec![1]),
        ],
        2,
        2,
    );
    let pruning = FrontierConfig {
        k: 1,
        delta: f64::INFINITY,
        score_alpha: 0.0,
        column_order: Some(vec![3, 2, 1, 0]),
        merge_indistinguishable: true,
        bp_score_iterations: 0,
    };
    let mut merged = FrontierDecoder::from_sparse_dem(&dem, pruning).unwrap();
    let mut hand_built = FrontierDecoder::from_sparse_dem(
        &premerged,
        FrontierConfig {
            k: 1,
            delta: f64::INFINITY,
            score_alpha: 0.0,
            column_order: None,
            merge_indistinguishable: false,
            bp_score_iterations: 0,
        },
    )
    .unwrap();
    let mut wrong_order = FrontierDecoder::from_sparse_dem(
        &moved_to_last,
        FrontierConfig {
            k: 1,
            delta: f64::INFINITY,
            score_alpha: 0.0,
            column_order: None,
            merge_indistinguishable: false,
            bp_score_iterations: 0,
        },
    )
    .unwrap();

    let actual = merged.decode(&[0, 0]).unwrap();
    let expected = hand_built.decode(&[0, 0]).unwrap();
    assert_eq!(actual, expected);
    assert_eq!(actual.processed_columns, 3);

    // For D0=0,D1=1, prefix-only K=1 drops the needed A branch when A is
    // first. Moving A after B retains a path, so this specifically guards the
    // first-occurrence deadline position rather than only the merged count.
    assert!(merged.decode(&[0, 1]).is_err());
    assert!(hand_built.decode(&[0, 1]).is_err());
    assert_eq!(
        wrong_order.decode(&[0, 1]).unwrap().predicted,
        ObsMask::from_u64(1)
    );
}

#[test]
fn merge_requires_matching_observable_sets() {
    let dem = sparse_dem(vec![(0.2, vec![0], vec![0]), (0.3, vec![0], vec![1])], 1, 2);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, merged_exact_config()).unwrap();
    let result = decoder.decode(&[1]).unwrap();
    let enumerated = independent_enumeration(&dem, &[1]);

    assert_eq!(result.processed_columns, 2);
    assert_mass_maps_close(
        &result_mass_map(&result),
        &enumerated,
        1e-12,
        "different observable symptoms",
    );
}

#[test]
fn forced_and_probabilistic_identical_symptoms_stay_in_separate_layers() {
    let dem = sparse_dem(vec![(1.0, vec![0], vec![0]), (0.3, vec![0], vec![0])], 1, 1);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, merged_exact_config()).unwrap();

    for observed in [[0], [1]] {
        let result = decoder.decode(&observed).unwrap();
        let enumerated = independent_enumeration(&dem, &observed);
        assert_eq!(result.processed_columns, 1);
        assert_eq!(result.predicted.words(), independent_winner(&enumerated));
        assert_mass_maps_close(
            &result_mass_map(&result),
            &enumerated,
            1e-12,
            "forced and probabilistic layers",
        );
    }
}

#[test]
fn zero_probability_duplicate_is_dropped_before_merging() {
    let dem = sparse_dem(vec![(0.0, vec![0], vec![0]), (0.3, vec![0], vec![0])], 1, 1);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, merged_exact_config()).unwrap();
    let result = decoder.decode(&[1]).unwrap();
    let enumerated = independent_enumeration(&dem, &[1]);

    assert_eq!(result.processed_columns, 1);
    assert_mass_maps_close(
        &result_mass_map(&result),
        &enumerated,
        1e-12,
        "zero-probability duplicate",
    );
}

#[test]
fn merge_is_default_off_and_preserves_the_unmerged_floating_path() {
    let default_config = FrontierConfig::default();
    assert!(!default_config.merge_indistinguishable);
    let dem = sparse_dem(vec![(0.3, vec![0], vec![0]), (0.3, vec![0], vec![0])], 1, 1);
    let premerged = sparse_dem(vec![(0.3 * 0.7 * 2.0, vec![0], vec![0])], 1, 1);
    let mut default_decoder = FrontierDecoder::from_sparse_dem(&dem, default_config).unwrap();
    let mut explicit_off = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    let mut hand_merged = FrontierDecoder::from_sparse_dem(&premerged, exact_config()).unwrap();
    let default_result = default_decoder.decode(&[1]).unwrap();
    let explicit_result = explicit_off.decode(&[1]).unwrap();
    let hand_merged_result = hand_merged.decode(&[1]).unwrap();

    assert_eq!(default_result.processed_columns, 2);
    assert_eq!(explicit_result.processed_columns, 2);
    assert_eq!(
        default_result.log_evidence.to_bits(),
        explicit_result.log_evidence.to_bits()
    );
    assert_eq!(
        default_result.logical_masses[0].log_mass.to_bits(),
        explicit_result.logical_masses[0].log_mass.to_bits()
    );
    assert_ne!(
        default_result.log_evidence.to_bits(),
        hand_merged_result.log_evidence.to_bits()
    );
    assert_ne!(default_result, hand_merged_result);
    assert_eq!(hand_merged_result.processed_columns, 1);
}

#[test]
fn degeneracy_mass_beats_the_single_most_likely_error() {
    // For syndrome D0=1, q=0.30's lone L0-flipping fault is the most likely
    // single configuration: q(1-p)^2 = 0.192 versus p(1-p)(1-q) = 0.112.
    // The no-logical-flip coset nevertheless wins by degeneracy:
    // (1-q)2p(1-p) = 0.224 versus q((1-p)^2+p^2) = 0.204.
    let dem = sparse_dem(
        vec![
            (0.30, vec![0], vec![0]),
            (0.20, vec![0], vec![]),
            (0.20, vec![0], vec![]),
        ],
        1,
        1,
    );
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    let result = decoder.decode(&[1]).unwrap();

    assert!(result.predicted.is_zero());
    assert!((result.log_evidence.exp() - (0.224 + 0.204)).abs() < 1e-12);
    assert!((result.logical_masses[0].log_mass.exp() - 0.224).abs() < 1e-12);
    assert!((result.runner_up_gap.unwrap() - (0.224_f64 / 0.204).ln()).abs() < 1e-12);
}

#[test]
fn supports_wide_detectors_and_observables_without_truncation() {
    let dem = sparse_dem(vec![(0.2, vec![69], vec![69])], 70, 70);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    let mut syndrome = vec![0; 70];
    syndrome[69] = 7;

    let mask = decoder.decode_obs(&syndrome).unwrap();
    assert!(mask.get(69));
    assert_eq!(mask.count_ones(), 1);
    assert!(matches!(
        decoder.decode_to_observables(&syndrome),
        Err(DecoderError::InvalidConfiguration(_))
    ));
}

#[test]
fn wide_logical_ties_use_numeric_label_order() {
    let dem = sparse_dem(
        vec![(0.5, vec![0], vec![0]), (0.5, vec![0], vec![64])],
        1,
        65,
    );
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    let result = decoder.decode(&[1]).unwrap();

    assert_eq!(result.predicted, ObsMask::from_u64(1));
    assert_eq!(result.runner_up_gap, Some(0.0));
    assert_eq!(result.logical_masses[0].logical, ObsMask::from_u64(1));
}

#[test]
fn rejects_duplicate_detector_and_observable_indices() {
    let duplicate_detector = sparse_dem(vec![(0.1, vec![0, 0], vec![])], 1, 0);
    let detector_error =
        FrontierDecoder::from_sparse_dem(&duplicate_detector, exact_config()).unwrap_err();
    let detector_message = detector_error.to_string();
    assert!(detector_message.contains("mechanism 0"));
    assert!(detector_message.contains("detector index 0"));

    let duplicate_observable = sparse_dem(vec![(0.1, vec![], vec![1, 1])], 0, 2);
    let observable_error =
        FrontierDecoder::from_sparse_dem(&duplicate_observable, exact_config()).unwrap_err();
    let observable_message = observable_error.to_string();
    assert!(observable_message.contains("mechanism 0"));
    assert!(observable_message.contains("observable index 1"));

    let parsed_error =
        FrontierDecoder::from_dem_str("error(0.1) D0 D0\n", FrontierConfig::default()).unwrap_err();
    assert!(parsed_error.to_string().contains("detector index 0"));
}

#[test]
fn forced_only_detector_requires_its_deterministic_syndrome() {
    let dem = sparse_dem(vec![(1.0, vec![0], vec![])], 1, 0);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();

    let matching = decoder.decode(&[1]).unwrap();
    assert!(matching.predicted.is_zero());
    assert_eq!(matching.log_evidence.to_bits(), 0.0_f64.to_bits());
    assert_eq!(matching.processed_columns, 0);
    assert!(decoder.decode(&[0]).is_err());
}

#[test]
fn forced_syndrome_shifts_shared_probabilistic_detector() {
    let dem = sparse_dem(vec![(1.0, vec![0], vec![0]), (0.2, vec![0], vec![1])], 1, 2);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();

    let probabilistic_skip = decoder.decode(&[1]).unwrap();
    assert_eq!(probabilistic_skip.predicted, ObsMask::from_u64(1));
    assert!((probabilistic_skip.log_evidence.exp() - 0.8).abs() < 1e-12);

    let probabilistic_take = decoder.decode(&[0]).unwrap();
    assert_eq!(probabilistic_take.predicted, ObsMask::from_u64(3));
    assert!((probabilistic_take.log_evidence.exp() - 0.2).abs() < 1e-12);
}

#[test]
fn forced_logical_flip_seeds_the_winning_label() {
    let dem = sparse_dem(vec![(1.0, vec![], vec![1]), (0.1, vec![], vec![0])], 0, 2);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    let result = decoder.decode(&[]).unwrap();

    assert_eq!(result.predicted, ObsMask::from_u64(2));
    assert!((result.logical_masses[0].log_mass.exp() - 0.9).abs() < 1e-12);
}

#[test]
fn truly_empty_dem_decodes_to_empty_label_with_unit_evidence() {
    let dem = sparse_dem(Vec::new(), 0, 0);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    let result = decoder.decode(&[]).unwrap();

    assert!(result.predicted.is_zero());
    assert_eq!(result.log_evidence.to_bits(), 0.0_f64.to_bits());
    assert_eq!(result.logical_masses.len(), 1);
}

#[test]
fn interleaved_syndromes_do_not_retain_decode_state() {
    let dem = sparse_dem(vec![(0.2, vec![0], vec![0]), (0.3, vec![1], vec![1])], 2, 2);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();

    let first_a = decoder.decode(&[1, 0]).unwrap();
    let b = decoder.decode(&[0, 1]).unwrap();
    let second_a = decoder.decode(&[1, 0]).unwrap();

    assert_eq!(first_a, second_a);
    assert_eq!(first_a.predicted, ObsMask::from_u64(1));
    assert_eq!(b.predicted, ObsMask::from_u64(2));
    assert_ne!(first_a.predicted, b.predicted);
}

#[test]
fn default_batch_decode_matches_individual_shots() {
    let dem = sparse_dem(vec![(0.2, vec![0], vec![0]), (0.3, vec![1], vec![1])], 2, 2);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    let expected = vec![
        decoder.decode_to_observables(&[1, 0]).unwrap(),
        decoder.decode_to_observables(&[0, 1]).unwrap(),
        decoder.decode_to_observables(&[1, 1]).unwrap(),
    ];

    let batched = decoder
        .decode_batch_to_observables(&[1, 0, 0, 1, 1, 1], 3, 2)
        .unwrap();
    assert_eq!(batched, expected);
}

#[test]
fn observable_only_mechanism_has_both_terminal_labels() {
    let dem = sparse_dem(vec![(0.2, vec![], vec![0])], 0, 1);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    let result = decoder.decode(&[]).unwrap();

    assert!(result.predicted.is_zero());
    assert_eq!(result.logical_masses.len(), 2);
    assert!((result.logical_masses[0].log_mass.exp() - 0.8).abs() < 1e-12);
    assert!((result.logical_masses[1].log_mass.exp() - 0.2).abs() < 1e-12);
    assert!(result.log_evidence.abs() < 1e-12);
}

#[test]
fn fails_loud_for_untouched_and_unachievable_syndromes() {
    let untouched_dem = sparse_dem(vec![(0.2, vec![0], vec![])], 2, 0);
    let mut untouched = FrontierDecoder::from_sparse_dem(&untouched_dem, exact_config()).unwrap();
    let untouched_error = untouched.decode(&[0, 1]).unwrap_err();
    assert!(untouched_error.to_string().contains("unexplainable"));

    let parity_locked_dem = sparse_dem(vec![(0.2, vec![0, 1], vec![])], 2, 0);
    let mut parity_locked =
        FrontierDecoder::from_sparse_dem(&parity_locked_dem, exact_config()).unwrap();
    let impossible_error = parity_locked.decode(&[1, 0]).unwrap_err();
    assert!(impossible_error.to_string().contains("pruning parameters"));
}

#[test]
fn overpruning_can_remove_the_only_eventually_feasible_prefix() {
    let dem = sparse_dem(
        vec![(0.4, vec![0], vec![]), (0.1, vec![0, 1], vec![0])],
        2,
        1,
    );
    let tight = FrontierConfig {
        k: 1,
        delta: 0.01,
        score_alpha: 0.0,
        column_order: None,
        merge_indistinguishable: false,
        bp_score_iterations: 0,
    };
    let mut overpruned = FrontierDecoder::from_sparse_dem(&dem, tight).unwrap();
    assert!(overpruned.decode(&[0, 1]).is_err());

    let mut default_decoder =
        FrontierDecoder::from_sparse_dem(&dem, FrontierConfig::default()).unwrap();
    assert_eq!(
        default_decoder.decode(&[0, 1]).unwrap().predicted,
        ObsMask::from_u64(1)
    );
}

#[test]
fn width_and_delta_pruning_can_change_the_logical_answer() {
    let dem = sparse_dem(
        vec![
            (0.20, vec![0], vec![]),
            (0.20, vec![0], vec![]),
            (0.30, vec![0], vec![0]),
        ],
        1,
        1,
    );
    let mut exact = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    assert!(exact.decode(&[1]).unwrap().predicted.is_zero());

    let mut greedy = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            k: 1,
            delta: f64::INFINITY,
            score_alpha: 0.0,
            column_order: None,
            merge_indistinguishable: false,
            bp_score_iterations: 0,
        },
    )
    .unwrap();
    assert_eq!(greedy.decode(&[1]).unwrap().predicted, ObsMask::from_u64(1));

    let mut delta_pruned = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            k: usize::MAX,
            delta: 0.1,
            score_alpha: 0.0,
            column_order: None,
            merge_indistinguishable: false,
            bp_score_iterations: 0,
        },
    )
    .unwrap();
    assert_eq!(
        delta_pruned.decode(&[1]).unwrap().predicted,
        ObsMask::from_u64(1)
    );
}

#[test]
fn transitions_count_every_candidate_branch_evaluation() {
    // Column 0 evaluates two branches from the initial state and retains two
    // distinct logical labels. Column 1 evaluates two branches from each of
    // those states, so the hand-derived total is 2 + 2*2 = 6.
    let dem = sparse_dem(vec![(0.2, vec![], vec![0]), (0.3, vec![], vec![1])], 0, 2);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    let result = decoder.decode(&[]).unwrap();

    assert_eq!(result.transitions, 6);
    assert_eq!(result.dropped_states, 0);
    assert_eq!(
        result.dropped_log_mass.to_bits(),
        f64::NEG_INFINITY.to_bits()
    );
    assert_eq!(result.status, FrontierStatus::Exact);
}

#[test]
fn width_pruning_accounts_for_the_discarded_state_and_mass() {
    // The only column creates skip/take states with masses 0.75 and 0.25.
    // K=1 keeps the former and drops exactly the latter.
    let dem = sparse_dem(vec![(0.25, vec![], vec![0])], 0, 1);
    let mut decoder = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            k: 1,
            delta: f64::INFINITY,
            score_alpha: 0.0,
            column_order: None,
            merge_indistinguishable: false,
            bp_score_iterations: 0,
        },
    )
    .unwrap();
    let result = decoder.decode(&[]).unwrap();

    assert_eq!(result.transitions, 2);
    assert_eq!(result.dropped_states, 1);
    assert!((result.dropped_log_mass.exp() - 0.25).abs() < 1e-15);
    assert_eq!(
        result.status,
        FrontierStatus::Pruned {
            k_capped: true,
            delta_pruned: false,
        }
    );
}

#[test]
fn delta_pruning_reports_its_status_flag() {
    // The two masses are 0.75 and 0.25, whose log-score gap ln(3) exceeds the
    // 0.5 window. With unlimited K, Delta alone drops the take state.
    let dem = sparse_dem(vec![(0.25, vec![], vec![0])], 0, 1);
    let mut decoder = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            k: usize::MAX,
            delta: 0.5,
            score_alpha: 0.0,
            column_order: None,
            merge_indistinguishable: false,
            bp_score_iterations: 0,
        },
    )
    .unwrap();
    let result = decoder.decode(&[]).unwrap();

    assert_eq!(result.dropped_states, 1);
    assert_eq!(
        result.status,
        FrontierStatus::Pruned {
            k_capped: false,
            delta_pruned: true,
        }
    );
}

#[test]
fn one_prune_call_can_trigger_both_pruning_flags() {
    // The first p=0.5 column retains two equal-mass states. The p=0.1 column
    // then produces masses 0.45, 0.45, 0.05, 0.05. K=3 removes the fourth;
    // within those first three, Delta=0.5 removes the third, so both mechanisms
    // discard one of the four hand-derived candidates.
    let dem = sparse_dem(vec![(0.5, vec![], vec![0]), (0.1, vec![], vec![1])], 0, 2);
    let mut decoder = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            k: 3,
            delta: 0.5,
            score_alpha: 0.0,
            column_order: None,
            merge_indistinguishable: false,
            bp_score_iterations: 0,
        },
    )
    .unwrap();
    let result = decoder.decode(&[]).unwrap();

    assert_eq!(result.dropped_states, 2);
    assert_eq!(
        result.status,
        FrontierStatus::Pruned {
            k_capped: true,
            delta_pruned: true,
        }
    );
}

#[test]
fn suffix_compatibility_changes_the_greedy_survivor() {
    // After column 0, prefix-only scoring prefers skip (mass 0.6) over take
    // (mass 0.4). For observed D0=1, the future p=0.1 column gives residual
    // compatibility rho=0.1 after skip and rho=0.9 after take. At alpha=0.8:
    //   skip: ln(0.6) + 0.8 ln(0.1) = -2.353
    //   take: ln(0.4) + 0.8 ln(0.9) = -1.001
    // Taking column 0 then skipping column 1 is the correct logical class;
    // prefix-only K=1 instead keeps skip and must take logical-flipping column 1.
    let dem = sparse_dem(vec![(0.4, vec![0], vec![]), (0.1, vec![0], vec![0])], 1, 1);
    let mut prefix_only = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            k: 1,
            delta: f64::INFINITY,
            score_alpha: 0.0,
            column_order: None,
            merge_indistinguishable: false,
            bp_score_iterations: 0,
        },
    )
    .unwrap();
    let mut suffix_scored = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            k: 1,
            delta: f64::INFINITY,
            score_alpha: 0.8,
            column_order: None,
            merge_indistinguishable: false,
            bp_score_iterations: 0,
        },
    )
    .unwrap();

    assert_eq!(
        prefix_only.decode(&[1]).unwrap().predicted,
        ObsMask::from_u64(1)
    );
    assert!(suffix_scored.decode(&[1]).unwrap().predicted.is_zero());
}

#[test]
fn decoding_is_bitwise_deterministic_including_a_tie() {
    let dem = sparse_dem(vec![(0.5, vec![0], vec![0]), (0.5, vec![0], vec![1])], 1, 2);
    let mut first_decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    let first = first_decoder.decode(&[1]).unwrap();
    let second = first_decoder.decode(&[1]).unwrap();
    let mut fresh_decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    let fresh = fresh_decoder.decode(&[1]).unwrap();

    assert_eq!(first, second);
    assert_eq!(first, fresh);
    assert_eq!(first.log_evidence.to_bits(), second.log_evidence.to_bits());
    assert_eq!(first.log_evidence.to_bits(), fresh.log_evidence.to_bits());
    assert_eq!(first.predicted, ObsMask::from_u64(1));
    assert_eq!(first.runner_up_gap, Some(0.0));
}

#[test]
fn works_through_observable_decoder_trait_object() {
    let dem = sparse_dem(vec![(0.2, vec![0], vec![0])], 1, 1);
    let decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    let mut boxed: Box<dyn ObservableDecoder> = Box::new(decoder);

    assert_eq!(boxed.decode_obs(&[1]).unwrap(), ObsMask::from_u64(1));
}

#[test]
fn parses_a_stim_dem_string() {
    let dem_text = "\
        detector(0, 0, 0) D0\n\
        detector(1, 0, 0) D1\n\
        logical_observable L0\n\
        error(0.1) D0 D1 L0\n\
        error(0.2) D1\n";
    let mut decoder = FrontierDecoder::from_dem_str(dem_text, exact_config()).unwrap();

    assert_eq!(decoder.decode_obs(&[1, 1]).unwrap(), ObsMask::from_u64(1));
}

#[test]
fn validates_probabilities_indices_order_and_pruning_configuration() {
    assert_eq!(
        FrontierConfig::default().score_alpha.to_bits(),
        0.8_f64.to_bits()
    );
    assert_eq!(FrontierConfig::default().bp_score_iterations, 0);
    let probability_one_dem = sparse_dem(vec![(1.0, vec![], vec![])], 0, 0);
    assert!(FrontierDecoder::from_sparse_dem(&probability_one_dem, exact_config()).is_ok());

    for probability in [1.000_000_1, 1.1, -0.1, f64::NAN, f64::INFINITY] {
        let dem = sparse_dem(vec![(probability, vec![], vec![])], 0, 0);
        assert!(matches!(
            FrontierDecoder::from_sparse_dem(&dem, FrontierConfig::default()),
            Err(DecoderError::InvalidConfiguration(_))
        ));
    }

    let zero_dem = sparse_dem(vec![(0.0, vec![], vec![])], 0, 0);
    let mut zero_decoder = FrontierDecoder::from_sparse_dem(&zero_dem, exact_config()).unwrap();
    let zero_result = zero_decoder.decode(&[]).unwrap();
    assert_eq!(zero_result.processed_columns, 0);
    assert_eq!(zero_result.log_evidence.to_bits(), 0.0_f64.to_bits());

    let invalid_zero_dem = sparse_dem(vec![(0.0, vec![0], vec![])], 0, 0);
    assert!(FrontierDecoder::from_sparse_dem(&invalid_zero_dem, exact_config()).is_err());

    let bad_detector = sparse_dem(vec![(0.1, vec![1], vec![])], 1, 0);
    assert!(FrontierDecoder::from_sparse_dem(&bad_detector, exact_config()).is_err());
    let bad_observable = sparse_dem(vec![(0.1, vec![], vec![1])], 0, 1);
    assert!(FrontierDecoder::from_sparse_dem(&bad_observable, exact_config()).is_err());

    let two_columns = sparse_dem(vec![(0.1, vec![], vec![]); 2], 0, 0);
    for bad_order in [vec![0], vec![0, 0], vec![0, 2]] {
        let config = FrontierConfig {
            column_order: Some(bad_order),
            ..FrontierConfig::default()
        };
        assert!(FrontierDecoder::from_sparse_dem(&two_columns, config).is_err());
    }

    let k_error = FrontierDecoder::from_sparse_dem(
        &zero_dem,
        FrontierConfig {
            k: 0,
            ..FrontierConfig::default()
        },
    )
    .unwrap_err();
    assert!(k_error.to_string().contains('k'));

    let negative_delta_error = FrontierDecoder::from_sparse_dem(
        &zero_dem,
        FrontierConfig {
            delta: -0.1,
            ..FrontierConfig::default()
        },
    )
    .unwrap_err();
    assert!(negative_delta_error.to_string().contains("delta"));

    let nan_delta_error = FrontierDecoder::from_sparse_dem(
        &zero_dem,
        FrontierConfig {
            delta: f64::NAN,
            ..FrontierConfig::default()
        },
    )
    .unwrap_err();
    assert!(nan_delta_error.to_string().contains("delta"));

    for score_alpha in [-0.1, f64::NAN, f64::INFINITY] {
        let alpha_error = FrontierDecoder::from_sparse_dem(
            &zero_dem,
            FrontierConfig {
                score_alpha,
                ..FrontierConfig::default()
            },
        )
        .unwrap_err();
        assert!(alpha_error.to_string().contains("score_alpha"));
    }
}

#[test]
fn rejects_wrong_syndrome_length() {
    let dem = sparse_dem(vec![(0.2, vec![0], vec![])], 1, 0);
    let mut decoder = FrontierDecoder::from_sparse_dem(&dem, exact_config()).unwrap();
    assert!(matches!(
        decoder.decode(&[]),
        Err(DecoderError::InvalidDimensions {
            expected: 1,
            actual: 0
        })
    ));
}

#[test]
fn bp_scores_change_the_greedy_survivor_without_changing_its_mass() {
    // For observed D0=1,D1=0, after column 0 the DEM-prior suffix estimate is:
    //   skip: 0.85 * (0.15 * 0.794)^0.8 = exp(-1.864...)
    //   take: 0.15 * (0.85 * 0.794)^0.8 = exp(-2.211...)
    // where 0.794 is the even-parity probability of future p=.15 and p=.08
    // faults on D1. Thus static K=1 keeps skip, which must later take columns 1
    // and 2 and produces L0=1.
    //
    // Five serial BP iterations give q1=0.101840... and q2=0.052612.... The
    // same row-moment arithmetic changes the suffix factors to about 0.0872
    // after skip and 0.7692 after take, narrowly reranking take above skip.
    // The surviving L0=0 path's DEM mass is still, independently,
    // 0.15 * 0.85 * 0.92 = 0.1173; BP supplies scores only.
    let dem = sparse_dem(
        vec![
            (0.15, vec![0], vec![]),
            (0.15, vec![0, 1], vec![0]),
            (0.08, vec![1], vec![]),
        ],
        2,
        1,
    );
    let syndrome = [1, 0];
    let bp_config = FrontierConfig {
        k: 1,
        delta: f64::INFINITY,
        score_alpha: 0.8,
        column_order: None,
        merge_indistinguishable: false,
        bp_score_iterations: 5,
    };
    let mut bp_scored = FrontierDecoder::from_sparse_dem(&dem, bp_config.clone()).unwrap();
    let mut dem_scored = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            bp_score_iterations: 0,
            ..bp_config
        },
    )
    .unwrap();

    let dem_result = dem_scored.decode(&syndrome).unwrap();
    let bp_result = bp_scored.decode(&syndrome).unwrap();
    assert_eq!(dem_result.predicted, ObsMask::from_u64(1));
    assert!(bp_result.predicted.is_zero());

    let enumerated = independent_enumeration(&dem, &syndrome);
    let expected_log_mass = enumerated
        .get(bp_result.predicted.words())
        .expect("the BP-retained label must exist in the exact enumeration");
    assert!((bp_result.logical_masses[0].log_mass.exp() - expected_log_mass.exp()).abs() <= 1e-12);
    assert!((bp_result.logical_masses[0].log_mass.exp() - 0.1173).abs() <= 1e-12);
}

#[test]
fn bp_scoring_is_bitwise_deterministic_across_reuse_and_fresh_construction() {
    let dem = sparse_dem(
        vec![
            (0.15, vec![0], vec![]),
            (0.15, vec![0, 1], vec![0]),
            (0.08, vec![1], vec![]),
        ],
        2,
        1,
    );
    let config = FrontierConfig {
        k: 1,
        delta: f64::INFINITY,
        score_alpha: 0.8,
        column_order: None,
        merge_indistinguishable: false,
        bp_score_iterations: 5,
    };
    let mut reused_decoder = FrontierDecoder::from_sparse_dem(&dem, config.clone()).unwrap();
    let first = reused_decoder.decode(&[1, 0]).unwrap();
    let second = reused_decoder.decode(&[1, 0]).unwrap();
    let mut fresh_decoder = FrontierDecoder::from_sparse_dem(&dem, config).unwrap();
    let fresh = fresh_decoder.decode(&[1, 0]).unwrap();

    assert!(first.bp_seconds >= 0.0 && second.bp_seconds >= 0.0 && fresh.bp_seconds >= 0.0);
    // Wall-clock telemetry is intentionally excluded from the bitwise semantic
    // comparison.
    assert_result_semantics_bitwise_equal(&first, &second);
    assert_result_semantics_bitwise_equal(&first, &fresh);
}
