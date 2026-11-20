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

fn same_result(expected: &TrellisResult, actual: &TrellisResult, flag_off: bool) {
    let TrellisResult {
        predicted,
        log_evidence,
        runner_up_gap,
        peak_retained_states,
        processed_columns,
        transitions,
        dropped_states,
        dropped_log_mass,
        bp_seconds,
        bp_runs,
        escalation_rungs_used,
        status,
        logical_masses,
    } = expected;
    assert_eq!(predicted, &actual.predicted);
    assert_eq!(log_evidence.to_bits(), actual.log_evidence.to_bits());
    assert_eq!(
        runner_up_gap.map(f64::to_bits),
        actual.runner_up_gap.map(f64::to_bits)
    );
    assert_eq!(*peak_retained_states, actual.peak_retained_states);
    assert_eq!(*processed_columns, actual.processed_columns);
    assert_eq!(*transitions, actual.transitions);
    assert_eq!(*dropped_states, actual.dropped_states);
    assert_eq!(*bp_runs, actual.bp_runs);
    assert_eq!(*escalation_rungs_used, actual.escalation_rungs_used);
    assert_eq!(*status, actual.status);
    assert!(bp_seconds.is_finite() && *bp_seconds >= 0.0);
    assert!(actual.bp_seconds.is_finite() && actual.bp_seconds >= 0.0);
    if *bp_runs == 0 {
        assert_eq!(bp_seconds.to_bits(), actual.bp_seconds.to_bits());
    }
    assert_eq!(logical_masses.len(), actual.logical_masses.len());
    for (left, right) in logical_masses.iter().zip(&actual.logical_masses) {
        let pecos_trellis::TrellisLogicalMass { logical, log_mass } = left;
        assert_eq!(*logical, right.logical);
        assert_eq!(log_mass.to_bits(), right.log_mass.to_bits());
    }
    if flag_off {
        assert!(dropped_log_mass.is_some());
        assert_eq!(actual.dropped_log_mass, None);
    } else {
        assert_eq!(
            dropped_log_mass.expect("enabled diagnostic").to_bits(),
            actual
                .dropped_log_mass
                .expect("enabled diagnostic")
                .to_bits()
        );
    }
}

fn same_attempt(expected: &TrellisDecodeAttempt, actual: &TrellisDecodeAttempt, flag_off: bool) {
    match (expected, actual) {
        (TrellisDecodeAttempt::Success(left), TrellisDecodeAttempt::Success(right)) => {
            same_result(left, right, flag_off);
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
            assert!(ab.is_finite() && bb.is_finite() && *ab >= 0.0 && *bb >= 0.0);
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
) -> TrellisDecodeAttempt {
    let mut progress = decoder.begin(shot).unwrap();
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
fn fixture_attempts_preserve_mass_bits_with_both_flags_no_path_and_bp_across_budgets() {
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
                dropped_mass: true,
                k,
                delta,
                score_alpha: alpha,
                bp_score_iterations: bp,
                merge_indistinguishable: merge,
                ..TrellisConfig::default()
            };
            let decoder = TrellisDecoder::from_sparse_dem(&dem, config.clone()).unwrap();
            let mut batch = decoder.clone();
            let mut bounded = decoder.into_resumable().unwrap();
            let mut skipped = TrellisDecoder::from_sparse_dem(
                &dem,
                TrellisConfig {
                    dropped_mass: false,
                    ..config
                },
            )
            .unwrap()
            .into_resumable()
            .unwrap();
            for budget in [1, 2, 3, 7, 64, usize::MAX] {
                for &syndrome in &fixture.syndromes {
                    let shot: Vec<_> = (0..fixture.num_detectors)
                        .map(|bit| u8::from(syndrome & (1_u128 << bit) != 0))
                        .collect();
                    let expected = batch.decode_attempt(&shot);
                    for flag_off in [false, true] {
                        successes +=
                            usize::from(matches!(expected, TrellisDecodeAttempt::Success(_)));
                        no_paths +=
                            usize::from(matches!(expected, TrellisDecodeAttempt::NoPath { .. }));
                        positive_work_no_paths += usize::from(
                            matches!(expected, TrellisDecodeAttempt::NoPath { transitions, dropped_states, .. } if transitions > 0 && dropped_states > 0),
                        );
                        let actual = drain(
                            if flag_off { &mut skipped } else { &mut bounded },
                            &shot,
                            budget,
                        );
                        same_attempt(&expected, &actual, flag_off);
                    }
                }
            }
            assert_eq!(
                batch.bp_refreshes(),
                bounded.bp_refreshes(),
                "{}",
                fixture.name
            );
            assert_eq!(
                batch.bp_refreshes(),
                skipped.bp_refreshes(),
                "{}",
                fixture.name
            );
        }
    }
    assert!(successes > 100 && no_paths > 10 && positive_work_no_paths > 0);
}

#[test]
fn large_wide_model_preserves_chunk_order_and_flag_on_diagnostics() {
    let dem = SparseDem {
        mechanisms: (0_u32..193)
            .map(|index| {
                (
                    0.01 + f64::from(index % 29) * 0.01,
                    vec![index % 5, if index % 2 == 0 { 70 } else { 130 }],
                    vec![index % 131],
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
            dropped_mass: true,
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
        let expected = batch.decode_attempt(&[0; 131]);
        if let TrellisDecodeAttempt::Success(result) = &expected {
            saw_pruning |= result.dropped_states > 0;
        }
        same_attempt(&expected, &drain(&mut bounded, &[0; 131], budget), false);
    }
    assert!(saw_pruning);
}

#[test]
fn protocol_failures_cancel_reuse_and_preparation_faults_preserve_state() {
    let base = TrellisDecoder::from_dem_str(
        "error(0.2) D0 L0\nerror(0.1) D0 D1\nerror(0.3) D1",
        TrellisConfig {
            dropped_mass: true,
            ..TrellisConfig::default()
        },
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
        &batch.decode_attempt(&[1, 0]),
        &drain(&mut bounded, &[1, 0], 1),
        false,
    );
    same_attempt(
        &batch.decode_attempt(&[0]),
        &drain(&mut bounded, &[0], 1),
        false,
    );
    same_attempt(
        &batch.decode_attempt(&[0, 1]),
        &drain(&mut bounded, &[0, 1], 2),
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
    let mut batch = TrellisDecoder::from_sparse_dem(
        &dem,
        TrellisConfig {
            dropped_mass: true,
            ..TrellisConfig::default()
        },
    )
    .unwrap();
    let mut bounded = batch.clone().into_resumable().unwrap();
    for shot in [&[0][..], &[1][..]] {
        same_attempt(
            &batch.decode_attempt(shot),
            &drain(&mut bounded, shot, 1),
            false,
        );
    }
    let integer = TrellisDecoder::from_sparse_dem(
        &dem,
        TrellisConfig {
            dropped_mass: true,
            metric_mode: MetricMode::MaxLogInt,
            ..TrellisConfig::default()
        },
    )
    .unwrap();
    // A rejected conversion hands back the unchanged decoder.
    let mut integer_reference = integer.clone();
    let Err((error, mut integer)) = integer.into_resumable() else {
        panic!("integer metric must be rejected");
    };
    assert!(matches!(error, DecoderError::InvalidConfiguration(_)));
    assert!(error.to_string().contains("MaxLogInt"), "{error}");
    for shot in [&[0][..], &[1][..]] {
        same_attempt(
            &integer_reference.decode_attempt(shot),
            &integer.decode_attempt(shot),
            false,
        );
    }
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
    let nary = TrellisDecoder::from_factor_model(
        &model,
        TrellisConfig {
            dropped_mass: true,
            ..TrellisConfig::default()
        },
    )
    .unwrap();
    let mut nary_reference = nary.clone();
    let Err((error, mut nary)) = nary.into_resumable() else {
        panic!("N-ary kernel must be rejected");
    };
    assert!(matches!(error, DecoderError::InvalidConfiguration(_)));
    assert!(error.to_string().contains("N-ary"), "{error}");
    same_attempt(
        &nary_reference.decode_attempt(&[1]),
        &nary.decode_attempt(&[1]),
        false,
    );
}
