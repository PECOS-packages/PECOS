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
use pecos_frontier::{FrontierConfig, FrontierDecoder, FrontierResult};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
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

fn exact_config() -> FrontierConfig {
    FrontierConfig {
        k: usize::MAX,
        delta: f64::INFINITY,
        column_order: None,
    }
}

fn independent_logaddexp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
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
        if detectors == observed {
            masses
                .entry(logical)
                .and_modify(|mass| *mass = independent_logaddexp(*mass, log_mass))
                .or_insert(log_mass);
        }
    }
    masses
}

fn independent_winner(masses: &BTreeMap<Vec<u64>, f64>) -> Vec<u64> {
    masses
        .iter()
        .min_by(|(left_label, left_mass), (right_label, right_mass)| {
            right_mass
                .total_cmp(left_mass)
                .then_with(|| left_label.cmp(right_label))
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

#[test]
fn unpruned_matches_independent_brute_force_on_seeded_random_dems() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x4652_4f4e_5449_4552);

    for case_index in 0..33 {
        let mechanism_count = 4 + case_index % 11;
        let num_detectors = rng.random_range(1..=10);
        let num_observables = rng.random_range(1..=3);
        let mut mechanisms = Vec::with_capacity(mechanism_count);
        let mut sampled_subset = Vec::with_capacity(mechanism_count);
        let mut observed = vec![0_u8; num_detectors];

        for mechanism_index in 0..mechanism_count {
            let probability = rng.random_range(0.01..0.4);
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

            let taken = rng.random_bool(0.35);
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
    assert!((result.log_evidence.exp() - 0.224).abs() < 1e-12);
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
        column_order: None,
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
            column_order: None,
        },
    )
    .unwrap();
    assert_eq!(greedy.decode(&[1]).unwrap().predicted, ObsMask::from_u64(1));

    let mut delta_pruned = FrontierDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            k: usize::MAX,
            delta: 0.1,
            column_order: None,
        },
    )
    .unwrap();
    assert_eq!(
        delta_pruned.decode(&[1]).unwrap().predicted,
        ObsMask::from_u64(1)
    );
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
    for probability in [1.0, 1.1, -0.1, f64::NAN, f64::INFINITY] {
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

    let one_column = sparse_dem(vec![(0.1, vec![], vec![])], 0, 0);
    for config in [
        FrontierConfig {
            k: 0,
            ..FrontierConfig::default()
        },
        FrontierConfig {
            delta: -0.1,
            ..FrontierConfig::default()
        },
        FrontierConfig {
            delta: f64::NAN,
            ..FrontierConfig::default()
        },
    ] {
        let mut decoder = FrontierDecoder::from_sparse_dem(&one_column, config).unwrap();
        let error = decoder.decode(&[]).unwrap_err();
        assert!(error.to_string().contains("pruning parameters"));
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
