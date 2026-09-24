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
use pecos_trellis::factor::{Factor, FactorModel, Outcome};
use pecos_trellis::{
    MetricMode, TrellisConfig, TrellisDecodeAttempt, TrellisDecoder, TrellisLogicalMass,
    TrellisResult, TrellisStatus,
};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::BTreeMap;

fn assert_result_bits(actual: &TrellisResult, expected: &TrellisResult) {
    // Exhaustive destructuring makes new result fields require a comparison.
    let TrellisResult {
        predicted,
        log_evidence,
        runner_up_gap,
        peak_retained_states,
        processed_columns,
        transitions,
        dropped_states,
        dropped_log_mass,
        bp_seconds: _,
        escalation_rungs_used,
        status,
        logical_masses,
    } = actual;
    assert_eq!(predicted, &expected.predicted);
    assert_eq!(log_evidence.to_bits(), expected.log_evidence.to_bits());
    assert_eq!(
        runner_up_gap.map(f64::to_bits),
        expected.runner_up_gap.map(f64::to_bits)
    );
    assert_eq!(*peak_retained_states, expected.peak_retained_states);
    assert_eq!(*processed_columns, expected.processed_columns);
    assert_eq!(*transitions, expected.transitions);
    assert_eq!(*dropped_states, expected.dropped_states);
    assert_eq!(
        dropped_log_mass.to_bits(),
        expected.dropped_log_mass.to_bits()
    );
    assert_eq!(*escalation_rungs_used, expected.escalation_rungs_used);
    assert_eq!(*status, expected.status);
    assert_eq!(logical_masses.len(), expected.logical_masses.len());
    for (TrellisLogicalMass { logical, log_mass }, expected) in
        logical_masses.iter().zip(&expected.logical_masses)
    {
        assert_eq!(logical, &expected.logical);
        assert_eq!(log_mass.to_bits(), expected.log_mass.to_bits());
    }
}

fn assert_attempt_bits(actual: &TrellisDecodeAttempt, expected: &TrellisDecodeAttempt) {
    match (actual, expected) {
        (TrellisDecodeAttempt::Success(actual), TrellisDecodeAttempt::Success(expected)) => {
            assert_result_bits(actual, expected);
        }
        (
            TrellisDecodeAttempt::NoPath {
                error: actual,
                transitions: actual_transitions,
                bp_seconds: _,
            },
            TrellisDecodeAttempt::NoPath {
                error: expected,
                transitions: expected_transitions,
                bp_seconds: _,
            },
        ) => {
            assert_eq!(actual.to_string(), expected.to_string());
            assert_eq!(actual_transitions, expected_transitions);
        }
        (TrellisDecodeAttempt::Error(actual), TrellisDecodeAttempt::Error(expected)) => {
            assert_eq!(actual.to_string(), expected.to_string());
        }
        _ => panic!("attempt variants differ"),
    }
}

fn check_sequence(build: impl Fn() -> TrellisDecoder, shots: &[Vec<u8>], pruned: bool) {
    let mut reused = build();
    let mut successes = 0;
    let mut saw_pruning = false;
    for (index, shot) in shots.iter().enumerate() {
        let actual = reused.decode_attempt(shot);
        let expected = build().decode_attempt(shot);
        assert_attempt_bits(&actual, &expected);
        match &actual {
            TrellisDecodeAttempt::Success(result) => {
                successes += 1;
                saw_pruning |= matches!(result.status, TrellisStatus::Pruned { .. });
                if !pruned {
                    assert_eq!(result.status, TrellisStatus::Exact);
                }
            }
            TrellisDecodeAttempt::NoPath { .. } => {}
            TrellisDecodeAttempt::Error(_) => assert_eq!(index, 5),
        }
        if index == 3 {
            assert!(matches!(actual, TrellisDecodeAttempt::NoPath { .. }));
        }
        if index == 5 {
            assert!(matches!(actual, TrellisDecodeAttempt::Error(_)));
        }
    }
    assert!(successes >= 3, "successes must surround the error paths");
    assert_eq!(saw_pruning, pruned);
}

#[test]
fn cross_shot_reuse_matches_fresh_decoders_bitwise() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5245_5553_455f_4450);
    for case in 0..12 {
        // Offset the small active model to exercise nonzero transpose origins
        // and unused detector words on both sides of the scoring span.
        let offset = [0_u32, 65, 129][case % 3];
        let active_count = rng.random_range(3..=6_u32);
        let num_detectors = (offset + active_count + 65) as usize;
        let mut mechanisms = Vec::new();
        let mut factors = Vec::new();
        for detector in offset..offset + active_count {
            let probability = rng.random_range(0.05..0.25);
            let observable = rng.random_range(0..2_u32);
            mechanisms.push((probability, vec![detector], vec![observable]));
            // A duplicate exercises binary merging when it is enabled.
            mechanisms.push((probability, vec![detector], vec![observable]));
            factors.push(Factor {
                outcomes: vec![
                    Outcome {
                        probability: 0.75 - probability,
                        detectors: vec![],
                        observables: vec![],
                    },
                    Outcome {
                        probability,
                        detectors: vec![detector],
                        observables: vec![observable],
                    },
                    Outcome {
                        probability: 0.25,
                        detectors: vec![detector],
                        observables: vec![1 - observable],
                    },
                ],
            });
        }
        // Cross-detector mechanisms keep suffix rows active across columns.
        for _ in 0..4 {
            let detectors: Vec<_> = (offset..offset + active_count)
                .filter(|_| rng.random_bool(0.5))
                .collect();
            let observables = vec![rng.random_range(0..2_u32)];
            let probability = rng.random_range(0.02..0.2);
            mechanisms.push((probability, detectors.clone(), observables.clone()));
            factors.push(Factor {
                outcomes: vec![
                    Outcome {
                        probability: 1.0 - probability,
                        detectors: vec![],
                        observables: vec![],
                    },
                    Outcome {
                        probability,
                        detectors,
                        observables,
                    },
                ],
            });
        }
        let dem = SparseDem {
            mechanisms,
            detector_coords: BTreeMap::new(),
            num_detectors,
            num_observables: 2,
        };
        let model = FactorModel::new(factors, num_detectors, 2).unwrap();
        let a = vec![0; num_detectors];
        let mut b = a.clone();
        b[offset as usize] = 1;
        let mut c = b.clone();
        c[(offset + 1) as usize] = 1;
        let mut no_path = a.clone();
        no_path[num_detectors - 1] = 1;
        let shots = [a.clone(), b.clone(), c, no_path, a, vec![0], b];

        for metric_mode in [MetricMode::LogSumExpFloat, MetricMode::MaxLogInt] {
            for pruned in [false, true] {
                for bp_score_iterations in [0, 5] {
                    for merge_indistinguishable in [false, true] {
                        // Max-log rejects mass-summing mechanism merging.
                        if metric_mode == MetricMode::MaxLogInt && merge_indistinguishable {
                            continue;
                        }
                        let config = TrellisConfig {
                            k: if pruned { 2 } else { usize::MAX },
                            delta: if pruned {
                                2.0
                            } else if metric_mode == MetricMode::MaxLogInt {
                                // Integer mode requires finite delta; this
                                // exceeds every score spread in these models.
                                1_000_000.0
                            } else {
                                f64::INFINITY
                            },
                            score_alpha: 0.8,
                            bp_score_iterations,
                            merge_indistinguishable,
                            metric_mode,
                            ..TrellisConfig::default()
                        };
                        check_sequence(
                            || TrellisDecoder::from_sparse_dem(&dem, config.clone()).unwrap(),
                            &shots,
                            pruned,
                        );
                        if bp_score_iterations == 0 && !merge_indistinguishable {
                            check_sequence(
                                || {
                                    TrellisDecoder::from_factor_model(&model, config.clone())
                                        .unwrap()
                                },
                                &shots,
                                pruned,
                            );
                        }
                    }
                }
            }
        }
    }
}
