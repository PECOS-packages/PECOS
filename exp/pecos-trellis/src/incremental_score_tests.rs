// Copyright 2026 The PECOS Developers
// SPDX-License-Identifier: Apache-2.0

use super::*;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

fn config() -> TrellisConfig {
    TrellisConfig {
        incremental_score: true,
        dropped_mass: true,
        ..TrellisConfig::default()
    }
}

fn sampled(dem: &SparseDem, rng: &mut Xoshiro256PlusPlus) -> Vec<u8> {
    let mut shot = vec![0; dem.num_detectors];
    for (probability, detectors, _) in &dem.mechanisms {
        if rng.random::<f64>() < *probability {
            for &detector in detectors {
                shot[detector as usize] ^= 1;
            }
        }
    }
    shot
}

pub(super) fn assert_bits(a: &TrellisResult, b: &TrellisResult) {
    let mut a = a.clone();
    let mut b = b.clone();
    a.bp_seconds = 0.0;
    b.bp_seconds = 0.0;
    assert_eq!(a, b);
    assert_eq!(a.log_evidence.to_bits(), b.log_evidence.to_bits());
    assert_eq!(
        a.runner_up_gap.map(f64::to_bits),
        b.runner_up_gap.map(f64::to_bits)
    );
    assert_eq!(
        a.dropped_log_mass.map(f64::to_bits),
        b.dropped_log_mass.map(f64::to_bits)
    );
    for (a, b) in a.logical_masses.iter().zip(&b.logical_masses) {
        assert_eq!(a.log_mass.to_bits(), b.log_mass.to_bits());
    }
}

#[test]
fn row_lists_match_brute_force_epochs() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5eed);
    for n in [4, 70, 140, 270] {
        let mut dem = SparseDem {
            mechanisms: (0..24)
                .map(|_| {
                    (
                        rng.random_range(0.01..0.4),
                        (0..n).filter(|_| rng.random_bool(0.1)).collect(),
                        vec![0],
                    )
                })
                .collect(),
            num_detectors: n as usize,
            num_observables: 1,
            detector_coords: BTreeMap::new(),
        };
        dem.mechanisms.insert(0, (1.0, vec![n - 1], vec![]));
        let decoder = TrellisDecoder::from_sparse_dem(&dem, config()).unwrap();
        let Kernel::Binary(columns) = &decoder.model.kernel else {
            panic!("binary");
        };
        let row_key = |r: &SuffixRow| (r.word, r.bit, r.epoch);
        for (c, column) in columns.iter().enumerate() {
            let previous = c
                .checked_sub(1)
                .map_or(&[][..], |i| &columns[i].suffix_compatibility);
            let current = &column.suffix_compatibility;
            let unchanged = |r: &SuffixRow, rows: &[SuffixRow]| {
                rows.iter().any(|other| row_key(r) == row_key(other))
            };
            let before: Vec<_> = previous
                .iter()
                .filter(|r| !unchanged(r, current))
                .map(row_key)
                .collect();
            let after: Vec<_> = current
                .iter()
                .filter(|r| !unchanged(r, previous))
                .map(row_key)
                .collect();
            let actual = &decoder.model.score_updates[c];
            assert_eq!(
                before,
                actual.before.iter().map(row_key).collect::<Vec<_>>(),
                "column {c}"
            );
            assert_eq!(
                after,
                actual.after.iter().map(row_key).collect::<Vec<_>>(),
                "column {c}"
            );
        }
        let summed = TrellisDecoder::from_sparse_dem(&dem, TrellisConfig::default()).unwrap();
        assert!(matches!(summed.model.kernel, Kernel::Binary(_)));
    }
}

#[test]
fn random_no_drop_results_are_bit_identical() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5eed);
    for offset in [0, 63, 128] {
        for _ in 0..20 {
            let dem = SparseDem {
                mechanisms: (0..10)
                    .map(|_| {
                        (
                            rng.random_range(0.01..0.4),
                            (offset..offset + 4)
                                .filter(|_| rng.random_bool(0.4))
                                .collect(),
                            (0..2).filter(|_| rng.random_bool(0.4)).collect(),
                        )
                    })
                    .collect(),
                num_detectors: (offset + 4) as usize,
                num_observables: 2,
                detector_coords: BTreeMap::new(),
            };
            let cfg = TrellisConfig {
                k: 1024,
                delta: f64::INFINITY,
                ..config()
            };
            let mut carried = TrellisDecoder::from_sparse_dem(&dem, cfg.clone()).unwrap();
            let mut summed = TrellisDecoder::from_sparse_dem(
                &dem,
                TrellisConfig {
                    incremental_score: false,
                    ..cfg
                },
            )
            .unwrap();
            for _ in 0..8 {
                let shot = sampled(&dem, &mut rng);
                let a = carried.decode(&shot).unwrap();
                let b = summed.decode(&shot).unwrap();
                assert_eq!(a.dropped_states, 0);
                assert_eq!(b.dropped_states, 0);
                assert_bits(&a, &b);
            }
        }
    }
}

#[test]
fn committed_model_pruned_agreement_with_and_without_bp() {
    let dem = SparseDem::from_dem_str(include_str!(
        "../../../examples/surface_code_circuits/surface_code_d5_z_stim.dem"
    ))
    .unwrap();
    for bp_score_iterations in [0, 5] {
        let cfg = TrellisConfig {
            k: 16,
            delta: 100.0,
            bp_score_iterations,
            ..config()
        };
        let mut carried = TrellisDecoder::from_sparse_dem(&dem, cfg.clone()).unwrap();
        let mut summed = TrellisDecoder::from_sparse_dem(
            &dem,
            TrellisConfig {
                incremental_score: false,
                ..cfg
            },
        )
        .unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5eed);
        let mut dropped = 0;
        for index in 0..8 {
            let shot = sampled(&dem, &mut rng);
            let a = carried.decode(&shot).unwrap();
            let b = summed.decode(&shot).unwrap();
            assert_eq!(a.status, b.status, "shot {index}, BP {bp_score_iterations}");
            assert_eq!(
                a.predicted, b.predicted,
                "shot {index}, BP {bp_score_iterations}"
            );
            assert!(
                (a.log_evidence - b.log_evidence).abs() <= 1e-6,
                "shot {index}, BP {bp_score_iterations}"
            );
            dropped += a.dropped_states;
        }
        assert!(dropped > 0);
    }
}

#[test]
fn nonfinite_suffix_is_rejected_only_when_incremental() {
    let dem = SparseDem::from_dem_str("error(0.1) D0 L0\nerror(1e-17) D0\n").unwrap();
    let summed = TrellisDecoder::from_sparse_dem(&dem, TrellisConfig::default()).unwrap();
    let Kernel::Binary(columns) = &summed.model.kernel else {
        panic!("binary");
    };
    let row = &columns[0].suffix_compatibility[0];
    assert_eq!(
        summed.model.suffix_values.probabilities[row.epoch as usize].one,
        f64::NEG_INFINITY
    );
    let error = TrellisDecoder::from_sparse_dem(&dem, config()).unwrap_err();
    assert!(matches!(error, DecoderError::InvalidConfiguration(_)));
    for text in [
        "incremental_score",
        "detector 0",
        "1 - 2p rounds to 1",
        "-inf",
        "leave incremental_score off",
    ] {
        assert!(error.to_string().contains(text), "{error}");
    }
    // BP's clamp makes the values used by the walk finite.
    let mut bp = TrellisDecoder::from_sparse_dem(
        &dem,
        TrellisConfig {
            bp_score_iterations: 5,
            ..config()
        },
    )
    .unwrap();
    bp.decode(&[1]).unwrap();
}

#[test]
fn resumable_prepared_attempts_and_skipped_scoring() {
    let dem = SparseDem::from_dem_str("error(1) D130\nerror(0.1) D0 L0\nerror(0.2) D0 D70\nerror(0.3) D70 D130 L1\nerror(0.15) D130\n").unwrap();
    for bp_score_iterations in [0, 3] {
        let cfg = TrellisConfig {
            k: 2,
            delta: 12.0,
            bp_score_iterations,
            ..config()
        };
        let mut batch = TrellisDecoder::from_sparse_dem(&dem, cfg.clone()).unwrap();
        let mut split = TrellisDecoder::from_sparse_dem(&dem, cfg)
            .unwrap()
            .into_resumable()
            .unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5eed);
        for _ in 0..12 {
            let shot = sampled(&dem, &mut rng);
            let expected = batch.decode(&shot).unwrap();
            let mut state = split.begin(&shot).unwrap();
            while !state.ready_to_finish {
                state = split.step(1).unwrap();
            }
            let TrellisDecodeAttempt::Success(actual) = split.finish().unwrap() else {
                panic!("success");
            };
            assert_bits(&expected, &actual);
            batch.prepare(&shot).unwrap();
            for params in [
                PruneParams {
                    k: usize::MAX,
                    delta: f64::INFINITY,
                },
                PruneParams { k: 2, delta: 12.0 },
            ] {
                let TrellisDecodeAttempt::Success(actual) = batch.attempt(params) else {
                    panic!("success");
                };
                if params.k == usize::MAX {
                    assert_eq!(
                        batch.scratch.float_progress.frontier.parent.suffix_sums,
                        [] as [f64; 0]
                    );
                } else {
                    assert_bits(&expected, &actual);
                }
            }
        }
    }
    for incremental_score in [false, true] {
        let mut decoder = TrellisDecoder::from_sparse_dem(
            &dem,
            TrellisConfig {
                score_alpha: 0.0,
                incremental_score,
                ..config()
            },
        )
        .unwrap();
        decoder.decode(&vec![0; dem.num_detectors]).unwrap();
        assert_eq!(
            decoder.scratch.float_progress.frontier.parent.suffix_sums,
            [] as [f64; 0]
        );
    }
}

#[test]
fn rejects_integer_and_nary_but_accepts_binary_factors() {
    let error = TrellisConfig {
        metric_mode: MetricMode::MaxLogInt,
        ..config()
    }
    .validate()
    .unwrap_err();
    assert!(matches!(error, DecoderError::InvalidConfiguration(_)));
    assert!(error.to_string().contains("incremental_score"));
    assert!(error.to_string().contains("LogSumExpFloat"));
    let outcome = |probability, observables| Outcome {
        probability,
        detectors: vec![],
        observables,
    };
    let nary = FactorModel::new(
        vec![factor::Factor {
            outcomes: vec![
                outcome(0.5, vec![]),
                outcome(0.3, vec![0]),
                outcome(0.2, vec![1]),
            ],
        }],
        0,
        2,
    )
    .unwrap();
    let error = TrellisDecoder::from_factor_model(&nary, config()).unwrap_err();
    assert!(matches!(error, DecoderError::InvalidConfiguration(_)));
    assert!(error.to_string().contains("incremental_score"));
    assert!(
        error
            .to_string()
            .contains("disable incremental_score for N-ary")
    );
    let binary = FactorModel::new(
        vec![factor::Factor {
            outcomes: vec![outcome(0.8, vec![]), outcome(0.2, vec![0])],
        }],
        0,
        1,
    )
    .unwrap();
    TrellisDecoder::from_factor_model(&binary, config())
        .unwrap()
        .decode(&[])
        .unwrap();
}

#[test]
fn carried_storage_preserves_first_arrival_and_retention() {
    // Distinct payloads deliberately test the generic storage contract independently
    // of mathematically equal suffix sums in a decoder's equal-key runs.
    for width in 0..=7 {
        for count in [16, 600, 1100] {
            let mut frontier = FrontierScratch::<f64> {
                stride: width,
                detector_words: width,
                ..Default::default()
            };
            for i in 0..count {
                for _ in 0..width {
                    frontier
                        .branches
                        .words
                        .push(if i % 2 == 0 { 0 } else { u64::MAX });
                }
                frontier.branches.masses.push(-1.0);
                frontier.branches.suffix_sums.push(f64::from(i));
            }
            let mut reference = frontier.clone();
            frontier.merge(logaddexp);
            reference.merge_reference(logaddexp);
            assert_eq!(frontier.parent.suffix_sums, reference.parent.suffix_sums);
            assert_eq!(
                frontier.parent.suffix_sums,
                if width == 0 {
                    vec![0.0]
                } else {
                    vec![0.0, 1.0]
                }
            );
            frontier.retained = (0..frontier.parent.masses.len())
                .map(|i| i == frontier.parent.masses.len() - 1)
                .collect();
            reference.retained.clone_from(&frontier.retained);
            frontier.retain_in_place();
            reference.retain();
            assert_eq!(frontier.parent.suffix_sums, reference.parent.suffix_sums);
            frontier.reset(&[], &[], &[], 0.0);
            assert_eq!(frontier.parent.suffix_sums, [] as [f64; 0]);
            assert_eq!(frontier.branches.suffix_sums, [] as [f64; 0]);
        }
    }
}

#[test]
fn bp_ladder_and_parallel_batch_carry_scores() {
    let dem = SparseDem::from_dem_str("error(0.4) D0\nerror(0.1) D0 D1 L0").unwrap();
    for bp_score_iterations in [0, 3] {
        let mut results = Vec::new();
        for incremental_score in [false, true] {
            let config = BpTrellisConfig {
                k: 16,
                delta: 0.01,
                score_alpha: 0.1,
                bp_score_iterations,
                incremental_score,
                merge_indistinguishable: false,
                ordering: TrellisOrdering::TimeOrder,
                escalation: vec![bp_trellis::EscalationRung {
                    k: 16,
                    delta: 100.0,
                }],
                ..BpTrellisConfig::default()
            };
            let mut decoder = BpTrellisDecoder::from_sparse_dem(&dem, config).unwrap();
            let result = decoder.decode(&[0, 1]).unwrap();
            assert_eq!(result.escalation_rungs_used, 1);
            results.push(result);
        }
        assert_bits(&results[0], &results[1]);
        let cfg = TrellisConfig {
            k: 2,
            bp_score_iterations,
            ..config()
        };
        let mut decoder = TrellisDecoder::from_sparse_dem(&dem, cfg).unwrap();
        let shots = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]];
        let batch = decoder.decode_batch(&shots, 2).unwrap();
        for (shot, result) in shots.iter().zip(batch) {
            let TrellisDecodeAttempt::Success(result) = result else {
                panic!("success");
            };
            assert_bits(&decoder.decode(shot).unwrap(), &result);
        }
    }
}

#[test]
fn bp_constructor_honours_incremental_score() {
    let dem = SparseDem::from_dem_str("error(0.1) D0 L0\nerror(1e-17) D0").unwrap();
    let config = BpTrellisConfig {
        bp_score_iterations: 0,
        ..BpTrellisConfig::default()
    };
    BpTrellisDecoder::from_sparse_dem(&dem, config.clone()).unwrap();
    let error = BpTrellisDecoder::from_sparse_dem(
        &dem,
        BpTrellisConfig {
            incremental_score: true,
            ..config
        },
    )
    .unwrap_err();
    assert!(matches!(error, DecoderError::InvalidConfiguration(_)));
    assert!(error.to_string().contains("incremental_score"), "{error}");
}

#[test]
fn carried_nan_scores_are_pruned() {
    // One state whose single active row has a NaN value, as a NaN BP posterior gives.
    let mut frontier = FrontierScratch::<f64> {
        stride: 1,
        detector_words: 1,
        ..Default::default()
    };
    frontier.parent.words.push(0);
    frontier.parent.masses.push(0.0);
    frontier.parent.suffix_sums.push(f64::NAN);
    let mut values = SuffixValues::new(1, None);
    values.probabilities[0].zero = f64::NAN;
    values.probabilities[0].one = f64::NAN;
    let rows = [SuffixRow {
        word: 0,
        epoch: 0,
        bit: 0,
    }];
    let result = prune(
        &mut frontier,
        1,
        1.0,
        1.0,
        SuffixCompatibility {
            rows: &rows,
            values: &values,
            incremental: true,
        },
        &[0],
        DecodeMode::Full,
    );
    assert_eq!(frontier.parent.masses, [] as [f64; 0]);
    assert_eq!(frontier.parent.suffix_sums, [] as [f64; 0]);
    assert_eq!(result.dropped_states, 1);
}
