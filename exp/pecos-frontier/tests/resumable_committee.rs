// Copyright 2026 The PECOS Developers
// SPDX-License-Identifier: Apache-2.0

use pecos_frontier::{
    CommitteeStatus, DecoderError, FrontierBoundedProgress, FrontierCommittee,
    FrontierCommitteeResult, FrontierConfig, FrontierResumableCommittee, SparseDem,
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

fn same_result(
    mut expected: FrontierCommitteeResult,
    mut actual: FrontierCommitteeResult,
    omitted: bool,
) {
    assert!(expected.selected.bp_seconds.is_finite() && actual.selected.bp_seconds.is_finite());
    assert_eq!(expected.direction, actual.direction);
    assert_eq!(
        expected.forward.log_evidence.to_bits(),
        actual.forward.log_evidence.to_bits()
    );
    assert_eq!(
        expected.backward.log_evidence.to_bits(),
        actual.backward.log_evidence.to_bits()
    );
    assert_eq!(
        expected.selected.log_evidence.to_bits(),
        actual.selected.log_evidence.to_bits()
    );
    assert_eq!(
        expected.selected.runner_up_gap.map(f64::to_bits),
        actual.selected.runner_up_gap.map(f64::to_bits)
    );
    for (left, right) in expected
        .selected
        .logical_masses
        .iter()
        .zip(&actual.selected.logical_masses)
    {
        assert_eq!(left.logical, right.logical);
        assert_eq!(left.log_mass.to_bits(), right.log_mass.to_bits());
    }
    if expected.selected.bp_runs != 0 {
        assert!(expected.selected.bp_seconds >= 0.0 && actual.selected.bp_seconds >= 0.0);
        expected.selected.bp_seconds = 0.0;
        actual.selected.bp_seconds = 0.0;
    }
    if omitted {
        assert!(actual.selected.dropped_log_mass.is_nan());
        expected.selected.dropped_log_mass = f64::NAN;
    } else {
        assert_eq!(
            expected.selected.dropped_log_mass.to_bits(),
            actual.selected.dropped_log_mass.to_bits()
        );
    }
    assert_eq!(format!("{expected:?}"), format!("{actual:?}"));
}

fn drain(
    decoder: &mut FrontierResumableCommittee,
    shot: &[u8],
    budget: usize,
    omitted: bool,
) -> Result<FrontierCommitteeResult, DecoderError> {
    let mut progress = if omitted {
        decoder.begin_without_dropped_mass(shot)
    } else {
        decoder.begin(shot)
    }
    .unwrap();
    assert!(matches!(progress, FrontierBoundedProgress::Forward(_)));
    while progress != FrontierBoundedProgress::Ready {
        let before = progress;
        progress = decoder.step(budget).unwrap();
        match (before, progress) {
            (FrontierBoundedProgress::Forward(a), FrontierBoundedProgress::Forward(b))
            | (FrontierBoundedProgress::Backward(a), FrontierBoundedProgress::Backward(b)) => {
                assert!(b.completed_columns - a.completed_columns <= budget);
            }
            (FrontierBoundedProgress::Forward(a), FrontierBoundedProgress::Backward(_))
            | (FrontierBoundedProgress::Backward(a), FrontierBoundedProgress::Ready) => {
                assert!(a.ready_to_finish);
            }
            _ => panic!("phase order changed"),
        }
        assert_eq!(decoder.progress(), Some(progress));
    }
    assert_eq!(
        decoder.step(usize::MAX).unwrap(),
        FrontierBoundedProgress::Ready
    );
    let result = decoder.finish();
    assert_eq!(decoder.progress(), None);
    result
}

#[test]
fn all_fixture_committees_preserve_bits_direction_fallback_and_bp() {
    let mut one_leg_no_path = 0;
    let mut both_no_path = 0;
    let mut successes = 0;
    for data in [
        include_str!("fixtures/upstream_fixtures.json"),
        include_str!("fixtures/upstream_order_fixtures.json"),
    ] {
        let file: FixtureFile = serde_json::from_str(data).unwrap();
        for fixture in file.fixtures {
            let dem = SparseDem {
                mechanisms: fixture.mechanisms,
                detector_coords: BTreeMap::new(),
                num_detectors: fixture.num_detectors,
                num_observables: fixture.num_observables,
            };
            for bp in [0, 2] {
                let config = FrontierConfig {
                    k: fixture.pruned.k,
                    delta: fixture.pruned.delta,
                    score_alpha: 0.8,
                    column_order: fixture.forward_ordering.clone(),
                    merge_indistinguishable: false,
                    bp_score_iterations: bp,
                    ..FrontierConfig::default()
                };
                let base = FrontierCommittee::from_sparse_dem(&dem, config).unwrap();
                let mut batch = base.clone();
                let mut bounded = base.into_resumable();
                for budget in [1, 2, 3, 7, 64, usize::MAX] {
                    for &mask in &fixture.syndromes {
                        let shot: Vec<_> = (0..fixture.num_detectors)
                            .map(|bit| u8::from(mask & (1_u128 << bit) != 0))
                            .collect();
                        for omitted in [false, true] {
                            match (
                                batch.decode(&shot),
                                drain(&mut bounded, &shot, budget, omitted),
                            ) {
                                (Ok(expected), Ok(actual)) => {
                                    successes += 1;
                                    one_leg_no_path += usize::from(
                                        expected.forward.status == CommitteeStatus::NoPath
                                            || expected.backward.status == CommitteeStatus::NoPath,
                                    );
                                    same_result(expected, actual, omitted);
                                }
                                (Err(expected), Err(actual)) => {
                                    both_no_path += usize::from(matches!(
                                        expected,
                                        DecoderError::DecodingFailed(_)
                                    ));
                                    assert_eq!(format!("{expected:?}"), format!("{actual:?}"));
                                }
                                _ => panic!("bounded committee outcome changed"),
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(successes > 100 && one_leg_no_path > 0 && both_no_path > 0);
}

#[test]
fn protocol_guards_cancel_reuse_and_dimension_faults_preserve_resolution() {
    let base = FrontierCommittee::from_dem_str(
        "error(0.2) D0 L0\nerror(0.1) D0 D1\nerror(0.3) D1",
        FrontierConfig::default(),
    )
    .unwrap();
    let mut bounded = base.clone().into_resumable();
    let invalid = |error| assert!(matches!(error, DecoderError::InvalidConfiguration(_)));
    invalid(bounded.step(1).unwrap_err());
    invalid(bounded.finish().unwrap_err());
    bounded.begin(&[0, 0]).unwrap();
    let before = bounded.progress();
    invalid(bounded.begin(&[1, 1]).unwrap_err());
    invalid(bounded.step(0).unwrap_err());
    invalid(bounded.finish().unwrap_err());
    assert_eq!(bounded.progress(), before);
    bounded.step(1).unwrap();
    bounded.cancel();
    bounded.cancel();
    assert_eq!(bounded.progress(), None);
    let mut batch = base;
    for shot in [&[1, 0][..], &[0][..], &[0, 1][..]] {
        match (batch.decode(shot), drain(&mut bounded, shot, 1, false)) {
            (Ok(expected), Ok(actual)) => same_result(expected, actual, false),
            (Err(expected), Err(actual)) => {
                assert_eq!(format!("{expected:?}"), format!("{actual:?}"));
            }
            _ => panic!("reuse or fault resolution changed"),
        }
    }
}

#[test]
fn cancel_during_backward_phase_leaves_both_legs_reusable() {
    for bp_score_iterations in [0, 2] {
        let base = FrontierCommittee::from_dem_str(
            "error(0.2) D0 L0\nerror(0.1) D0 D1\nerror(0.3) D1\nerror(0.05) D1 D2 L1\nerror(0.1) D2",
            FrontierConfig {
                bp_score_iterations,
                ..FrontierConfig::default()
            },
        )
        .unwrap();
        let mut batch = base.clone();
        let mut bounded = base.into_resumable();
        bounded.begin(&[1, 1, 0]).unwrap();
        // Drain forward, transition, then stop part-way through backward.
        while !matches!(
            bounded.progress(),
            Some(FrontierBoundedProgress::Backward(_))
        ) {
            bounded.step(1).unwrap();
        }
        bounded.step(1).unwrap();
        let Some(FrontierBoundedProgress::Backward(partial)) = bounded.progress() else {
            panic!("a five-column backward leg is still pending after one column");
        };
        assert!(partial.completed_columns > 0 && !partial.ready_to_finish);
        bounded.cancel();
        assert_eq!(bounded.progress(), None);
        for shot in [&[0, 1, 1][..], &[1, 1, 0][..]] {
            for budget in [1, 2, usize::MAX] {
                match (batch.decode(shot), drain(&mut bounded, shot, budget, false)) {
                    (Ok(expected), Ok(actual)) => same_result(expected, actual, false),
                    (Err(expected), Err(actual)) => {
                        assert_eq!(format!("{expected:?}"), format!("{actual:?}"));
                    }
                    _ => panic!("cancelled backward leg changed a later window"),
                }
            }
        }
    }
}

#[test]
fn empty_and_forced_models_still_run_both_leg_phases() {
    for text in ["detector D0", "error(1) D0 L0\ndetector D1"] {
        let base = FrontierCommittee::from_dem_str(text, FrontierConfig::default()).unwrap();
        let mut batch = base.clone();
        let mut bounded = base.into_resumable();
        for shot in [&[0][..], &[1][..], &[1, 0][..]] {
            match (batch.decode(shot), drain(&mut bounded, shot, 1, false)) {
                (Ok(expected), Ok(actual)) => same_result(expected, actual, false),
                (Err(expected), Err(actual)) => {
                    assert_eq!(format!("{expected:?}"), format!("{actual:?}"));
                }
                _ => panic!("zero-column or preparation attempt changed"),
            }
        }
    }
}
