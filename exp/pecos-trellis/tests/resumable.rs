// Copyright 2026 The PECOS Developers
// SPDX-License-Identifier: Apache-2.0

use pecos_trellis::{
    DecoderError, MetricMode, SparseDem, TrellisConfig, TrellisDecodeAttempt, TrellisDecoder,
    TrellisResult, TrellisResumableDecoder,
};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Fixture {
    name: String,
    #[serde(default)]
    mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    num_detectors: usize,
    num_observables: usize,
    syndromes: Vec<u128>,
}

fn same_result(mut expected: TrellisResult, mut actual: TrellisResult, omitted: bool) {
    // BP clocks measure two separate executions. Every other float is compared
    // by bits, including signed zero and the full discarded-mass diagnostic.
    assert!(expected.bp_seconds.is_finite() && actual.bp_seconds.is_finite());
    if expected.bp_runs != 0 {
        assert!(expected.bp_seconds >= 0.0 && actual.bp_seconds >= 0.0);
        expected.bp_seconds = 0.0;
        actual.bp_seconds = 0.0;
    }
    assert_eq!(
        expected.log_evidence.to_bits(),
        actual.log_evidence.to_bits()
    );
    assert_eq!(
        expected.runner_up_gap.map(f64::to_bits),
        actual.runner_up_gap.map(f64::to_bits)
    );
    if omitted {
        assert!(actual.dropped_log_mass.is_nan());
        expected.dropped_log_mass = f64::NAN;
    } else {
        assert_eq!(
            expected.dropped_log_mass.to_bits(),
            actual.dropped_log_mass.to_bits()
        );
    }
    for (left, right) in expected.logical_masses.iter().zip(&actual.logical_masses) {
        assert_eq!(left.logical, right.logical);
        assert_eq!(left.log_mass.to_bits(), right.log_mass.to_bits());
    }
    // Debug has round-trippable float representations and includes every field.
    assert_eq!(format!("{expected:?}"), format!("{actual:?}"));
}

fn same_attempt(expected: TrellisDecodeAttempt, actual: TrellisDecodeAttempt, omitted: bool) {
    match (expected, actual) {
        (TrellisDecodeAttempt::Success(left), TrellisDecodeAttempt::Success(right)) => {
            same_result(left, right, omitted);
        }
        (
            TrellisDecodeAttempt::NoPath {
                error: a,
                transitions: at,
                dropped_states: ad,
                bp_seconds: ab,
            },
            TrellisDecodeAttempt::NoPath {
                error: b,
                transitions: bt,
                dropped_states: bd,
                bp_seconds: bb,
            },
        ) => {
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
            assert_eq!((at, ad), (bt, bd));
            assert!(ab.is_finite() && bb.is_finite() && ab >= 0.0 && bb >= 0.0);
        }
        (TrellisDecodeAttempt::Error(a), TrellisDecodeAttempt::Error(b)) => {
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
        }
        _ => panic!("resumable and batch attempt variants differ"),
    }
}

fn drain(
    decoder: &mut TrellisResumableDecoder,
    shot: &[u8],
    budget: usize,
    omitted: bool,
) -> TrellisDecodeAttempt {
    let mut progress = if omitted {
        decoder.begin_without_dropped_mass(shot)
    } else {
        decoder.begin(shot)
    }
    .unwrap();
    assert_eq!(decoder.progress(), Some(progress));
    while !progress.ready_to_finish {
        let before = progress.completed_columns;
        progress = decoder.step(budget).unwrap();
        assert!(progress.completed_columns - before <= budget);
    }
    assert_eq!(decoder.step(usize::MAX).unwrap(), progress);
    let result = decoder.finish().unwrap();
    assert_eq!(decoder.progress(), None);
    result
}

#[test]
fn fixture_attempts_preserve_full_mass_bits_no_path_and_bp_across_budgets() {
    let mut fixtures: BTreeMap<String, Vec<Fixture>> =
        serde_json::from_str(include_str!("fixtures/models.json")).unwrap();
    let fixtures = fixtures.remove("binary").unwrap();
    let mut successes = 0;
    let mut no_paths = 0;
    let mut positive_work_no_paths = 0;
    for fixture in fixtures {
        let dem = SparseDem {
            mechanisms: fixture.mechanisms,
            num_detectors: fixture.num_detectors,
            num_observables: fixture.num_observables,
            detector_coords: BTreeMap::new(),
        };
        for (k, delta, alpha, bp, merge) in [
            (1, 1e9, 0.8, 0, false),
            (16, 1.5, 0.0, 0, true),
            (4, 1e9, 0.8, 2, false),
        ] {
            let config = TrellisConfig {
                k,
                delta,
                score_alpha: alpha,
                bp_score_iterations: bp,
                merge_indistinguishable: merge,
                ..TrellisConfig::default()
            };
            let decoder = TrellisDecoder::from_sparse_dem(&dem, config).unwrap();
            let mut batch = decoder.clone();
            let mut bounded = decoder.into_resumable().unwrap();
            for budget in [1, 2, 3, 7, 64, usize::MAX] {
                for &syndrome in &fixture.syndromes {
                    let shot: Vec<_> = (0..fixture.num_detectors)
                        .map(|bit| u8::from(syndrome & (1_u128 << bit) != 0))
                        .collect();
                    for omitted in [false, true] {
                        let expected = batch.decode_attempt(&shot);
                        successes +=
                            usize::from(matches!(expected, TrellisDecodeAttempt::Success(_)));
                        no_paths +=
                            usize::from(matches!(expected, TrellisDecodeAttempt::NoPath { .. }));
                        positive_work_no_paths += usize::from(
                            matches!(expected, TrellisDecodeAttempt::NoPath { transitions, dropped_states, .. } if transitions > 0 && dropped_states > 0),
                        );
                        let actual = drain(&mut bounded, &shot, budget, omitted);
                        same_attempt(expected, actual, omitted);
                    }
                }
            }
            assert_eq!(
                batch.bp_refreshes(),
                bounded.bp_refreshes(),
                "{}",
                fixture.name
            );
        }
    }
    assert!(successes > 100 && no_paths > 10 && positive_work_no_paths > 0);
}

#[test]
fn large_wide_model_preserves_chunk_order_and_full_diagnostics() {
    let dem = SparseDem {
        mechanisms: (0..193)
            .map(|index| {
                (
                    0.01 + (index % 29) as f64 * 0.01,
                    vec![(index % 5) as u32, if index % 2 == 0 { 70 } else { 130 }],
                    vec![(index % 131) as u32],
                )
            })
            .collect(),
        num_detectors: 131,
        num_observables: 131,
        detector_coords: BTreeMap::new(),
    };
    let base = TrellisDecoder::from_sparse_dem(
        &dem,
        TrellisConfig {
            k: 32,
            delta: 1e9,
            score_alpha: 0.8,
            merge_indistinguishable: false,
            ..TrellisConfig::default()
        },
    )
    .unwrap();
    let mut saw_pruning = false;
    for budget in [1, 2, 3, 7, 64, usize::MAX] {
        let mut batch = base.clone();
        let mut bounded = base.clone().into_resumable().unwrap();
        let expected = batch.decode_attempt(&vec![0; 131]);
        if let TrellisDecodeAttempt::Success(result) = &expected {
            saw_pruning |= result.dropped_states > 0;
        }
        same_attempt(
            expected,
            drain(&mut bounded, &vec![0; 131], budget, false),
            false,
        );
    }
    assert!(saw_pruning);
}

#[test]
fn protocol_failures_cancel_reuse_and_preparation_faults_preserve_state() {
    let base = TrellisDecoder::from_dem_str(
        "error(0.2) D0 L0\nerror(0.1) D0 D1\nerror(0.3) D1",
        TrellisConfig::default(),
    )
    .unwrap();
    let mut bounded = base.clone().into_resumable().unwrap();
    let invalid = |error| assert!(matches!(error, DecoderError::InvalidConfiguration(_)));
    invalid(bounded.step(1).unwrap_err());
    invalid(bounded.finish().err().unwrap());
    bounded.begin(&[0, 0]).unwrap();
    let before = bounded.progress();
    invalid(bounded.begin(&[1, 1]).unwrap_err());
    invalid(bounded.step(0).unwrap_err());
    invalid(bounded.finish().err().unwrap());
    assert_eq!(bounded.progress(), before);
    bounded.step(1).unwrap();
    bounded.cancel();
    bounded.cancel();
    assert_eq!(bounded.progress(), None);
    let mut batch = base;
    same_attempt(
        batch.decode_attempt(&[1, 0]),
        drain(&mut bounded, &[1, 0], 1, false),
        false,
    );
    same_attempt(
        batch.decode_attempt(&[0]),
        drain(&mut bounded, &[0], 1, false),
        false,
    );
    same_attempt(
        batch.decode_attempt(&[0, 1]),
        drain(&mut bounded, &[0, 1], 2, false),
        false,
    );
}

#[test]
fn empty_kernel_and_unsupported_metric_have_explicit_outcomes() {
    use pecos_trellis::factor::{Factor, FactorModel, Outcome};
    let dem = SparseDem {
        mechanisms: vec![],
        num_detectors: 1,
        num_observables: 131,
        detector_coords: BTreeMap::new(),
    };
    let mut batch = TrellisDecoder::from_sparse_dem(&dem, TrellisConfig::default()).unwrap();
    let mut bounded = batch.clone().into_resumable().unwrap();
    for shot in [&[0][..], &[1][..]] {
        same_attempt(
            batch.decode_attempt(shot),
            drain(&mut bounded, shot, 1, false),
            false,
        );
    }
    let integer = TrellisDecoder::from_sparse_dem(
        &dem,
        TrellisConfig {
            metric_mode: MetricMode::MaxLogInt,
            ..TrellisConfig::default()
        },
    )
    .unwrap();
    assert!(matches!(
        integer.into_resumable().err().unwrap(),
        DecoderError::InvalidConfiguration(_)
    ));
    let model = FactorModel::new(
        vec![Factor {
            outcomes: vec![
                Outcome {
                    probability: 0.1,
                    detectors: vec![0],
                    observables: vec![0],
                },
                Outcome {
                    probability: 0.2,
                    detectors: vec![0],
                    observables: vec![],
                },
                Outcome {
                    probability: 0.7,
                    detectors: vec![],
                    observables: vec![],
                },
            ],
        }],
        1,
        1,
    )
    .unwrap();
    let nary = TrellisDecoder::from_factor_model(&model, TrellisConfig::default()).unwrap();
    assert!(matches!(
        nary.into_resumable().err().unwrap(),
        DecoderError::InvalidConfiguration(_)
    ));
}
