// Copyright 2026 The PECOS Developers
// SPDX-License-Identifier: Apache-2.0

//! Constructor-only legacy closing-mask oracle; never performs a DP decode.

use super::{
    IMPLICIT_ZERO_WORD, Kernel, SparseDem, TrellisConfig, TrellisDecoder, WORD_BITS,
    deadline_column_order,
};
use crate::factor::{Factor, FactorModel, Outcome};
use std::collections::BTreeMap;

fn contains(words: &[u64], detector: usize) -> bool {
    words[detector / WORD_BITS] & (1_u64 << (detector % WORD_BITS)) != 0
}

fn words_for_indices(width: usize, detectors: &[usize]) -> Vec<u64> {
    let mut words = vec![0; width.div_ceil(WORD_BITS)];
    for &detector in detectors {
        assert!(detector < width);
        words[detector / WORD_BITS] |= 1_u64 << (detector % WORD_BITS);
    }
    words
}

/// Reconstruct last-touch with the legacy *all-detectors* scan, independently
/// of the current-toggle/support-only scan the constructors use.
fn assert_legacy_masks(decoder: &TrellisDecoder) -> usize {
    let model = &decoder.model;
    let width = model.num_detectors;
    let words = model.detector_words;
    let supports: Vec<Vec<u64>> = match &model.kernel {
        Kernel::Binary(columns) => columns.iter().map(|c| c.detector_toggle.clone()).collect(),
        Kernel::Nary(columns) => columns
            .iter()
            .map(|column| {
                let mut support = vec![0; words];
                for outcome in &column.outcomes {
                    assert_eq!(outcome.detector_toggle.len(), words);
                    for (combined, &value) in support.iter_mut().zip(&outcome.detector_toggle) {
                        *combined |= value;
                    }
                }
                support
            })
            .collect(),
    };
    let mut last_touch = vec![None; width];
    let mut touched = vec![0; words];
    for (column, support) in supports.iter().enumerate() {
        assert_eq!(support.len(), words);
        for (combined, &value) in touched.iter_mut().zip(support) {
            *combined |= value;
        }
        for (detector, last) in last_touch.iter_mut().enumerate() {
            if contains(support, detector) {
                *last = Some(column);
            }
        }
    }
    assert_eq!(model.touched_detectors, touched);
    let mut active: Vec<u64> = model
        .forced_syndrome
        .iter()
        .zip(&touched)
        .map(|(&forced, &probabilistic)| forced & probabilistic)
        .collect();
    let mut already_closed = vec![0_u64; words];
    let mut previous_span = (0, words);
    for (column_index, support) in supports.iter().enumerate() {
        let mut expected_close = vec![0_u64; words];
        // This deliberately retains the original full last_touch scan.
        for (detector, &last) in last_touch.iter().enumerate() {
            if last == Some(column_index) {
                expected_close[detector / WORD_BITS] |= 1_u64 << (detector % WORD_BITS);
            }
        }
        for (word, ((value, &toggle), &closing)) in active
            .iter_mut()
            .zip(support)
            .zip(&expected_close)
            .enumerate()
        {
            assert_eq!(
                closing & !toggle,
                0,
                "closing detector must occur in current support"
            );
            assert_eq!(already_closed[word] & closing, 0, "detector closes twice");
            already_closed[word] |= closing;
            *value = (*value | toggle) & !closing;
        }
        let (actual_close, actual_active, has_closing, suffix) = match &model.kernel {
            Kernel::Binary(columns) => {
                let c = &columns[column_index];
                let first_word = active.iter().position(|&word| word != 0).unwrap_or(0);
                let end = active
                    .iter()
                    .rposition(|&word| word != 0)
                    .map_or(0, |i| i + 1);
                let source = |word: usize| {
                    word.checked_sub(previous_span.0)
                        .filter(|&index| index < previous_span.1)
                        .unwrap_or(IMPLICIT_ZERO_WORD)
                };
                let expected_output: Vec<_> = (first_word..end)
                    .map(|word| (source(word), support[word], active[word]))
                    .collect();
                let expected_closing: Vec<_> = expected_close
                    .iter()
                    .enumerate()
                    .filter(|&(_, &mask)| mask != 0)
                    .map(|(word, &mask)| (source(word), word, mask, support[word]))
                    .collect();
                assert_eq!(c.compact.first_word, first_word);
                assert_eq!(c.compact.parent_words, previous_span.1);
                assert_eq!(c.compact.output, expected_output);
                assert_eq!(c.compact.closing, expected_closing);
                previous_span = (first_word, end - first_word);
                (
                    &c.close_mask,
                    &c.active_mask,
                    c.has_closing_detectors,
                    &c.suffix_compatibility,
                )
            }
            Kernel::Nary(columns) => {
                let c = &columns[column_index];
                (
                    &c.close_mask,
                    &c.active_mask,
                    c.has_closing_detectors,
                    &c.suffix_compatibility,
                )
            }
        };
        assert_eq!(
            *actual_close, expected_close,
            "legacy close mask at column {column_index}"
        );
        assert_eq!(
            *actual_active, active,
            "legacy active mask at column {column_index}"
        );
        assert_eq!(has_closing, expected_close.iter().any(|&word| word != 0));
        let expected_rows: Vec<_> = (0..width).filter(|&d| contains(&active, d)).collect();
        let actual_rows: Vec<_> = suffix
            .iter()
            .map(|row| {
                let detector =
                    usize::try_from(row.word).unwrap() * WORD_BITS + usize::from(row.bit);
                let epoch = usize::try_from(row.epoch).unwrap();
                assert_eq!(model.suffix_epochs[epoch].detector, detector);
                detector
            })
            .collect();
        assert_eq!(
            actual_rows, expected_rows,
            "suffix scoring rows at column {column_index}"
        );
    }
    assert_eq!(already_closed, touched);
    assert!(active.iter().all(|&word| word == 0));
    supports.len()
}

fn config(order: Option<Vec<usize>>) -> TrellisConfig {
    TrellisConfig {
        column_order: order,
        merge_indistinguishable: false,
        bp_score_iterations: 0,
        ..TrellisConfig::default()
    }
}

fn wide_dem() -> SparseDem {
    SparseDem {
        mechanisms: vec![
            (0.0, vec![7, 129], vec![2]),
            (1.0, vec![0, 64, 130, 192], vec![65]),
            (0.1, vec![0, 63, 64], vec![0]),
            (0.2, vec![1, 64, 127], vec![64]),
            (1.0, vec![64], vec![1]),
            (0.3, vec![63, 65, 128, 129], vec![130]),
            (0.4, vec![], vec![64]),
            (0.2, vec![130], vec![65]),
            (0.0, vec![], vec![0]),
        ],
        detector_coords: BTreeMap::new(),
        num_detectors: 193,
        num_observables: 131,
    }
}

#[test]
fn binary_closing_oracle_wide_zero_forced_empty_and_reordered() {
    let dem = wide_dem();
    for order in [None, Some((0..dem.mechanisms.len()).rev().collect())] {
        let decoder = TrellisDecoder::from_sparse_dem(&dem, config(order)).unwrap();
        assert_eq!(assert_legacy_masks(&decoder), 5);
        assert_eq!(
            decoder.model.forced_syndrome,
            words_for_indices(193, &[0, 130, 192])
        );
        assert_eq!(
            decoder.model.forced_logical,
            words_for_indices(131, &[1, 65])
        );
        assert_eq!(
            decoder.model.touched_detectors,
            words_for_indices(193, &[0, 1, 63, 64, 65, 127, 128, 129, 130])
        );
    }
}

#[test]
fn binary_closing_oracle_empty_and_all_deterministic_models() {
    for (width, terms) in [
        (0, vec![]),
        (193, vec![]),
        (193, vec![(0.0, vec![0, 130], vec![0])]),
        (193, vec![(1.0, vec![64, 130], vec![0])]),
        (
            193,
            vec![(1.0, vec![64], vec![0]), (1.0, vec![64], vec![0])],
        ),
    ] {
        let dem = SparseDem {
            mechanisms: terms,
            detector_coords: BTreeMap::new(),
            num_detectors: width,
            num_observables: 1,
        };
        let decoder = TrellisDecoder::from_sparse_dem(&dem, config(None)).unwrap();
        assert_eq!(assert_legacy_masks(&decoder), 0);
    }
}

#[test]
fn binary_closing_oracle_many_word_boundaries_and_orders() {
    for width in [0, 1, 63, 64, 65, 127, 128, 129, 193] {
        let mechanisms = (0..24)
            .map(|i| {
                let mut ds = if width == 0 {
                    vec![]
                } else {
                    vec![
                        i % width,
                        (37 * i + 1) % width,
                        (67 * i + width - 1) % width,
                    ]
                };
                ds.sort_unstable();
                ds.dedup();
                let ds = ds.into_iter().map(|d| u32::try_from(d).unwrap()).collect();
                (
                    [0.0, 1.0, 0.125, 0.5, 0.875][i % 5],
                    ds,
                    vec![u32::try_from(i % 65).unwrap()],
                )
            })
            .collect();
        let dem = SparseDem {
            mechanisms,
            detector_coords: BTreeMap::new(),
            num_detectors: width,
            num_observables: 65,
        };
        let order = deadline_column_order(&dem).unwrap();
        for order in [None, Some(order), Some((0..24).rev().collect())] {
            let decoder = TrellisDecoder::from_sparse_dem(&dem, config(order)).unwrap();
            assert_legacy_masks(&decoder);
        }
    }
}

fn outcome(probability: f64, detectors: &[u32], observables: &[u32]) -> Outcome {
    Outcome {
        probability,
        detectors: detectors.to_vec(),
        observables: observables.to_vec(),
    }
}

#[test]
fn nary_closing_oracle_uses_outcome_union_and_ignores_zero_outcomes() {
    let model = FactorModel::new(
        vec![
            Factor {
                outcomes: vec![
                    outcome(1.0, &[0, 64, 130, 192], &[65]),
                    outcome(0.0, &[7], &[0]),
                ],
            },
            Factor {
                outcomes: vec![
                    outcome(0.5, &[], &[]),
                    outcome(0.3, &[0, 63], &[64]),
                    outcome(0.2, &[64, 129], &[130]),
                ],
            },
            Factor {
                outcomes: vec![
                    outcome(0.4, &[], &[]),
                    outcome(0.4, &[63, 65, 128], &[]),
                    outcome(0.2, &[130], &[65]),
                ],
            },
            Factor {
                outcomes: vec![
                    outcome(0.5, &[], &[]),
                    outcome(0.5, &[], &[0]),
                    outcome(0.0, &[7, 192], &[]),
                ],
            },
            Factor {
                outcomes: vec![outcome(1.0, &[64], &[1])],
            },
        ],
        193,
        131,
    )
    .unwrap();
    for order in [None, Some((0..model.factors().len()).rev().collect())] {
        let decoder = TrellisDecoder::from_factor_model(&model, config(order)).unwrap();
        assert!(matches!(decoder.model.kernel, Kernel::Nary(_)));
        assert_eq!(assert_legacy_masks(&decoder), 3);
        assert_eq!(
            decoder.model.forced_syndrome,
            words_for_indices(193, &[0, 130, 192])
        );
    }
}

#[test]
fn nary_closing_oracle_detector_free_outcomes_and_binary_delegate() {
    let nary = FactorModel::new(
        vec![Factor {
            outcomes: vec![
                outcome(0.5, &[], &[]),
                outcome(0.3, &[], &[0]),
                outcome(0.2, &[], &[130]),
            ],
        }],
        193,
        131,
    )
    .unwrap();
    let decoder = TrellisDecoder::from_factor_model(&nary, config(None)).unwrap();
    assert!(matches!(decoder.model.kernel, Kernel::Nary(_)));
    assert_eq!(assert_legacy_masks(&decoder), 1);
    let binary = FactorModel::try_from(&wide_dem()).unwrap();
    let decoder = TrellisDecoder::from_factor_model(&binary, config(None)).unwrap();
    assert!(matches!(decoder.model.kernel, Kernel::Binary(_)));
    assert_eq!(assert_legacy_masks(&decoder), 5);
}
