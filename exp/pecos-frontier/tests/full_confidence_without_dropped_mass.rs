// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

//! Only the discarded-prefix-mass diagnostic may differ in the opt-in API.

use pecos_frontier::{
    CommitteeDirection, FrontierCommittee, FrontierCommitteeResult, FrontierConfig,
    FrontierDecodeAttempt, FrontierDecoder, FrontierResult, MetricMode, SparseDem,
};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct FixtureFile {
    fixtures: Vec<Fixture>,
}

#[derive(Deserialize)]
struct Fixture {
    mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    num_detectors: usize,
    num_observables: usize,
    syndromes: Vec<u128>,
    pruned: Pruning,
    #[serde(default)]
    forward_ordering: Option<Vec<usize>>,
}

#[derive(Deserialize)]
struct Pruning {
    k: usize,
    delta: f64,
}

fn dem(fixture: &Fixture) -> SparseDem {
    SparseDem {
        mechanisms: fixture.mechanisms.clone(),
        detector_coords: BTreeMap::new(),
        num_detectors: fixture.num_detectors,
        num_observables: fixture.num_observables,
    }
}

fn dense(mask: u128, count: usize) -> Vec<u8> {
    (0..count).map(|i| ((mask >> i) & 1) as u8).collect()
}

fn config(k: usize, delta: f64, column_order: Option<Vec<usize>>, bp: usize) -> FrontierConfig {
    FrontierConfig {
        k,
        delta,
        score_alpha: 0.8,
        column_order,
        merge_indistinguishable: false,
        bp_score_iterations: bp,
        metric_mode: MetricMode::LogSumExpFloat,
        int_metric_scale: 1024,
    }
}

fn confidence_bits(result: &FrontierResult) -> u64 {
    let mut marginal = [f64::NEG_INFINITY; 2];
    for class in &result.logical_masses {
        let accumulator = &mut marginal[usize::from(class.logical.get(0))];
        *accumulator = if *accumulator == f64::NEG_INFINITY {
            class.log_mass
        } else if class.log_mass == f64::NEG_INFINITY {
            *accumulator
        } else {
            let high = accumulator.max(class.log_mass);
            high + (accumulator.min(class.log_mass) - high).exp().ln_1p()
        };
    }
    (marginal[1] - marginal[0]).abs().to_bits()
}

fn assert_retained_bits(full: &FrontierResult, skipped: &FrontierResult) {
    assert!(!full.dropped_log_mass.is_nan(), "default full diagnostic must remain available");
    assert!(skipped.dropped_log_mass.is_nan(), "opt-in diagnostic must be explicitly unavailable");
    assert_eq!(full.predicted, skipped.predicted);
    assert_eq!(full.log_evidence.to_bits(), skipped.log_evidence.to_bits());
    assert_eq!(full.runner_up_gap.map(f64::to_bits), skipped.runner_up_gap.map(f64::to_bits));
    assert_eq!(full.peak_retained_states, skipped.peak_retained_states);
    assert_eq!(full.processed_columns, skipped.processed_columns);
    assert_eq!(full.transitions, skipped.transitions);
    assert_eq!(full.dropped_states, skipped.dropped_states);
    assert_eq!(full.bp_runs, skipped.bp_runs);
    assert_eq!(full.escalation_rungs_used, skipped.escalation_rungs_used);
    assert_eq!(full.status, skipped.status);
    assert_eq!(full.logical_masses.len(), skipped.logical_masses.len());
    for (left, right) in full.logical_masses.iter().zip(&skipped.logical_masses) {
        assert_eq!(left.logical, right.logical);
        assert_eq!(left.log_mass.to_bits(), right.log_mass.to_bits());
    }
    assert_eq!(confidence_bits(full), confidence_bits(skipped));
    // BP wall-clock duration is not a numerical decoder output. BP run counts
    // and every retained numerical result above remain exactly equal.
    if full.bp_runs == 0 {
        assert_eq!(full.bp_seconds.to_bits(), skipped.bp_seconds.to_bits());
    }
}

fn assert_attempts(full: FrontierDecodeAttempt, skipped: FrontierDecodeAttempt) -> bool {
    match (full, skipped) {
        (FrontierDecodeAttempt::Success(full), FrontierDecodeAttempt::Success(skipped)) => {
            assert_retained_bits(&full, &skipped);
            true
        }
        (FrontierDecodeAttempt::NoPath { error: left, transitions: lt, dropped_states: ld, .. },
         FrontierDecodeAttempt::NoPath { error: right, transitions: rt, dropped_states: rd, .. }) => {
            assert_eq!(format!("{left:?}"), format!("{right:?}"));
            assert_eq!((lt, ld), (rt, rd));
            false
        }
        (FrontierDecodeAttempt::Error(left), FrontierDecodeAttempt::Error(right)) => {
            assert_eq!(format!("{left:?}"), format!("{right:?}"));
            false
        }
        _ => panic!("opt-in changed success/no-path/error provenance"),
    }
}

fn assert_committee(full: &FrontierCommitteeResult, skipped: &FrontierCommitteeResult) {
    assert_retained_bits(&full.selected, &skipped.selected);
    assert_eq!(full.direction, skipped.direction);
    assert_eq!(full.forward.status, skipped.forward.status);
    assert_eq!(full.backward.status, skipped.backward.status);
    assert_eq!(full.forward.log_evidence.to_bits(), skipped.forward.log_evidence.to_bits());
    assert_eq!(full.backward.log_evidence.to_bits(), skipped.backward.log_evidence.to_bits());
}

#[test]
fn all_fixture_retained_outputs_and_committee_arbitration_are_bitwise_equal() {
    let mut successes = 0;
    let mut backward = 0;
    let mut failed = 0;
    for source in [include_str!("fixtures/upstream_fixtures.json"), include_str!("fixtures/upstream_order_fixtures.json")] {
        let file: FixtureFile = serde_json::from_str(source).unwrap();
        for fixture in file.fixtures {
            let model = dem(&fixture);
            for (k, delta) in [(usize::MAX, f64::INFINITY), (fixture.pruned.k, fixture.pruned.delta)] {
                let options = config(k, delta, fixture.forward_ordering.clone(), 0);
                let mut full = FrontierCommittee::from_sparse_dem(&model, options.clone()).unwrap();
                let mut skipped = FrontierCommittee::from_sparse_dem(&model, options.clone()).unwrap();
                // Each leg is also checked independently, including the losing
                // leg's entire retained class vector, not only its summary.
                let order = options.column_order.clone().unwrap_or_else(|| (0..model.mechanisms.len()).collect());
                let mut engines = Vec::new();
                for column_order in [order.clone(), order.into_iter().rev().collect()] {
                    let mut options = options.clone();
                    options.column_order = Some(column_order);
                    engines.push((FrontierDecoder::from_sparse_dem(&model, options.clone()).unwrap(),
                                  FrontierDecoder::from_sparse_dem(&model, options).unwrap()));
                }
                for mask in &fixture.syndromes {
                    let observed = dense(*mask, model.num_detectors);
                    for (left, right) in &mut engines {
                        assert_attempts(left.decode_attempt(&observed), right.decode_attempt_without_dropped_mass(&observed));
                    }
                    match (full.decode(&observed), skipped.decode_without_dropped_mass(&observed)) {
                        (Ok(left), Ok(right)) => {
                            assert_committee(&left, &right);
                            successes += 1;
                            backward += usize::from(left.direction == CommitteeDirection::Backward);
                            // The default method remains full even after using
                            // the opt-in API on this same stateful committee.
                            let again = skipped.decode(&observed).unwrap();
                            assert_eq!(left, again);
                        }
                        (Err(left), Err(right)) => {
                            assert_eq!(format!("{left:?}"), format!("{right:?}"));
                            failed += 1;
                        }
                        _ => panic!("opt-in changed committee success/no-path outcome"),
                    }
                }
            }
        }
    }
    assert!(successes > 0 && backward > 0 && failed > 0);
}

#[test]
fn forced_zero_wide_labels_bp_and_invalid_input_preserve_retained_results() {
    let model = SparseDem {
        mechanisms: vec![(0.1, vec![0], vec![0]), (0.2, vec![1], vec![1]),
            (0.15, vec![0, 1], vec![130]), (0.03, vec![2], vec![0, 2]),
            (0.07, vec![1, 2], vec![2]), (0.4, vec![], vec![0]),
            (1.0, vec![2], vec![130]), (0.0, vec![0], vec![2])],
        detector_coords: BTreeMap::new(), num_detectors: 3, num_observables: 131,
    };
    for bp in [0, 2] {
        for k in [1, 2, 512] {
            let options = config(k, 1e9, None, bp);
            let mut left = FrontierDecoder::from_sparse_dem(&model, options.clone()).unwrap();
            let mut right = FrontierDecoder::from_sparse_dem(&model, options.clone()).unwrap();
            let mut full = FrontierCommittee::from_sparse_dem(&model, options.clone()).unwrap();
            let mut skipped = FrontierCommittee::from_sparse_dem(&model, options).unwrap();
            for mask in 0..8 {
                let observed = dense(mask, 3);
                assert_attempts(left.decode_attempt(&observed), right.decode_attempt_without_dropped_mass(&observed));
                match (full.decode(&observed), skipped.decode_without_dropped_mass(&observed)) {
                    (Ok(a), Ok(b)) => assert_committee(&a, &b),
                    (Err(a), Err(b)) => assert_eq!(format!("{a:?}"), format!("{b:?}")),
                    _ => panic!("different committee outcome"),
                }
            }
            assert_eq!(left.bp_refreshes(), right.bp_refreshes());
            assert_attempts(left.decode_attempt(&[0]), right.decode_attempt_without_dropped_mass(&[0]));
            let a = full.decode(&[0]).unwrap_err();
            let b = skipped.decode_without_dropped_mass(&[0]).unwrap_err();
            assert_eq!(format!("{a:?}"), format!("{b:?}"));
            // Recovery after an invalid call uses the same strict defaults.
            assert_attempts(left.decode_attempt(&[0, 0, 0]), right.decode_attempt_without_dropped_mass(&[0, 0, 0]));
        }
    }
}
