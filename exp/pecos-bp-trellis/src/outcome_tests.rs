// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

use super::*;
use serde::Deserialize;
use std::collections::BTreeMap;

const OVERPRUNING: &str = "error(0.4) D0\nerror(0.4) D1\nerror(0.1) D0 D1 D2 L0";

fn config(k: usize, rungs: &[(usize, f64)]) -> BpTrellisConfig {
    BpTrellisConfig {
        k,
        delta: f64::INFINITY,
        score_alpha: 0.0,
        bp_score_iterations: 0,
        merge_indistinguishable: false,
        ordering: TrellisOrdering::TimeOrder,
        escalation: rungs
            .iter()
            .map(|&(k, delta)| EscalationRung { k, delta })
            .collect(),
    }
}

fn no_path(outcome: BpTrellisOutcome) -> NoPathReport {
    let BpTrellisOutcome::NoPath(report) = outcome else {
        panic!("expected no-path")
    };
    report
}

fn counted(decoder: &mut BpTrellisDecoder, shot: &[u8], expected: usize) -> NoPathReport {
    let mut calls = 0;
    let outcome = decoder
        .decode_with_attempt(shot, |inner, params| {
            calls += 1;
            assert!(calls <= expected, "unexpected attempt call");
            inner.attempt(params)
        })
        .unwrap();
    assert_eq!(calls, expected, "number of actual attempt calls");
    no_path(outcome)
}

fn mapping(decoder: &mut BpTrellisDecoder, shot: &[u8], message: &str) {
    let report = no_path(decoder.decode_outcome(shot).unwrap());
    assert_eq!(
        report.clone().into_error().to_string(),
        decoder.decode(shot).unwrap_err().to_string()
    );
    assert_eq!(
        report.into_error().to_string(),
        DecoderError::DecodingFailed(message.into()).to_string()
    );
}

#[test]
fn residual_skips_all_attempts_and_preserves_forced_mask() {
    for (dem, shot, detector) in [
        ("error(1) D0 L0 L70", vec![0], 0),
        ("error(1) L0 L70\ndetector D1", vec![0, 1], 1),
    ] {
        let mut decoder =
            BpTrellisDecoder::from_dem_str(dem, config(1, &[(2, 100.0), (4, 100.0), (8, 100.0)]))
                .unwrap();
        let report = counted(&mut decoder, &shot, 0);
        assert_eq!(report.cause, NoPathCause::Residual { detector });
        assert_eq!(report.placeholder, ObsMask::from_words(&[1, 64]));
        assert_eq!(report.rungs_tried, 0);
        assert_eq!(report.transitions, 0);
        assert_eq!(report.bp_runs, 0);
        assert_eq!(report.bp_seconds.to_bits(), 0.0_f64.to_bits());
        mapping(
            &mut decoder,
            &shot,
            &format!(
                "syndrome is unexplainable: detector {detector} has a residual no mechanism can change"
            ),
        );
    }
}

#[test]
fn infeasible_base_skips_ladder() {
    let mut decoder = BpTrellisDecoder::from_dem_str(
        "error(0.2) D0 D1",
        config(1, &[(2, 100.0), (4, 100.0), (8, 100.0)]),
    )
    .unwrap();
    let report = counted(&mut decoder, &[1, 0], 1);
    assert_eq!(report.cause, NoPathCause::Infeasible);
    assert_eq!(report.rungs_tried, 0);
    assert_eq!(report.transitions, 2);
    assert!(report.placeholder.is_zero());
    mapping(
        &mut decoder,
        &[1, 0],
        "syndrome is unexplainable under the detector error model",
    );
}

#[test]
fn infeasible_rung_skips_remaining_rungs() {
    let mut decoder = BpTrellisDecoder::from_dem_str(
        "error(0.1) L0\nerror(0.2) D0 D1",
        config(1, &[(2, f64::INFINITY), (4, f64::INFINITY)]),
    )
    .unwrap();
    let report = counted(&mut decoder, &[1, 0], 2);
    assert_eq!(report.cause, NoPathCause::Infeasible);
    assert_eq!(report.rungs_tried, 1);
    mapping(
        &mut decoder,
        &[1, 0],
        "syndrome is unexplainable under the detector error model",
    );
}

#[test]
fn exhausted_sums_every_attempt_exactly() {
    let cfg = config(1, &[(1, f64::INFINITY), (2, f64::INFINITY)]);
    let dem = SparseDem::from_dem_str(OVERPRUNING).unwrap();
    let mut engine = TrellisDecoder::from_sparse_dem(&dem, cfg.trellis_config()).unwrap();
    engine.prepare(&[0, 0, 1]).unwrap();
    let expected: u64 = [1, 1, 2]
        .into_iter()
        .map(|k| {
            let TrellisDecodeAttempt::NoPath {
                transitions,
                dropped_states,
                ..
            } = engine.attempt(PruneParams {
                k,
                delta: f64::INFINITY,
            })
            else {
                panic!("no-path")
            };
            assert!(dropped_states > 0);
            transitions
        })
        .sum();
    let mut decoder = BpTrellisDecoder::from_sparse_dem(&dem, cfg).unwrap();
    let report = counted(&mut decoder, &[0, 0, 1], 3);
    assert_eq!(report.cause, NoPathCause::Exhausted);
    assert_eq!(report.rungs_tried, 2);
    assert_eq!(report.transitions, expected);
    mapping(
        &mut decoder,
        &[0, 0, 1],
        "syndrome is unexplainable at the given pruning parameters after 2 escalation rungs",
    );
}

fn decoded_mapping(dem: &str, cfg: BpTrellisConfig, shot: &[u8]) {
    let mut decoder = BpTrellisDecoder::from_dem_str(dem, cfg).unwrap();
    let BpTrellisOutcome::Decoded(result) = decoder.decode_outcome(shot).unwrap() else {
        panic!("rung must recover")
    };
    assert_eq!(result.escalation_rungs_used, 1);
    assert_eq!(decoder.decode(shot).unwrap(), result);
}

#[test]
fn narrower_rung_recovers() {
    let dem = "error(0.4) D1 D3 L0\nerror(0.3) D1 D2\nerror(0.05) D2 D3\nerror(0.3) D0 D1\nerror(0.05) D0 D1 D3\nerror(0.2) D3\nerror(0.05) D0 D1 D3 L0";
    let mut cfg = config(3, &[(2, 50.0)]);
    cfg.delta = 100.0;
    decoded_mapping(dem, cfg, &[1, 1, 1, 0]);
}

#[test]
fn rung_delta_is_applied() {
    let mut cfg = config(16, &[(16, 100.0)]);
    cfg.delta = 0.01;
    decoded_mapping("error(0.4) D0\nerror(0.1) D0 D1 L0", cfg, &[0, 1]);
}

#[test]
fn bp_refresh_is_reused_across_the_ladder() {
    #[derive(Deserialize)]
    struct Fixture {
        name: String,
        mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
        num_detectors: usize,
        num_observables: usize,
    }
    let fixtures: BTreeMap<String, Vec<Fixture>> =
        serde_json::from_str(include_str!("../tests/fixtures/models.json")).unwrap();
    let f = fixtures
        .values()
        .flatten()
        .find(|f| f.name == "bp_sensitive_ladder")
        .unwrap();
    let dem = SparseDem {
        mechanisms: f.mechanisms.clone(),
        num_detectors: f.num_detectors,
        num_observables: f.num_observables,
        detector_coords: BTreeMap::new(),
    };
    let cfg = BpTrellisConfig {
        k: 1,
        escalation: vec![EscalationRung {
            k: 16,
            delta: 100.0,
        }],
        merge_indistinguishable: false,
        ..BpTrellisConfig::default()
    };
    let mut decoder = BpTrellisDecoder::from_sparse_dem(&dem, cfg).unwrap();
    let mut shot = vec![0; f.num_detectors];
    shot[1] = 1;
    // Observe each attempt's preparation time directly, then compare the attached
    // result. The count witnesses discarded as well as used extra prepares.
    let before = decoder.inner.bp_refreshes();
    let mut times = Vec::new();
    let outcome = decoder
        .decode_with_attempt(&shot, |inner, params| {
            let attempt = inner.attempt(params);
            times.push(match &attempt {
                TrellisDecodeAttempt::Success(r) => r.bp_seconds.to_bits(),
                TrellisDecodeAttempt::NoPath { bp_seconds, .. } => bp_seconds.to_bits(),
                TrellisDecodeAttempt::Error(e) => panic!("{e}"),
            });
            attempt
        })
        .unwrap();
    let BpTrellisOutcome::Decoded(result) = outcome else {
        panic!("ladder must decode")
    };
    assert_eq!(result.escalation_rungs_used, 1);
    assert_eq!(decoder.inner.bp_refreshes() - before, 1);
    assert_eq!(result.bp_runs, 1);
    assert_eq!(times, vec![result.bp_seconds.to_bits(); 2]);
    // Also exercise the public entry point with this exact one-rung ladder.
    let before = decoder.inner.bp_refreshes();
    let BpTrellisOutcome::Decoded(result) = decoder.decode_outcome(&shot).unwrap() else {
        panic!("decoded")
    };
    assert_eq!(result.bp_runs, 1);
    assert_eq!(decoder.inner.bp_refreshes() - before, 1);
}

#[test]
fn rung_validation_has_no_dominance_rule() {
    let mut cfg = config(usize::MAX, &[(16, 100.0)]);
    assert!(
        cfg.validate()
            .unwrap_err()
            .to_string()
            .contains("exact base never prunes")
    );
    cfg.k = 8;
    cfg.escalation.push(EscalationRung { k: 0, delta: 50.0 });
    assert!(
        cfg.validate()
            .unwrap_err()
            .to_string()
            .contains("escalation[1]:")
    );
    cfg.escalation[1].k = 2;
    cfg.validate().unwrap();
    for delta in [f64::NAN, -1.0, f64::NEG_INFINITY] {
        cfg.escalation[1].delta = delta;
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("escalation[1]:")
        );
    }
}

#[test]
fn errors_stay_errors_with_ladder() {
    let cfg = config(1, &[(16, 100.0)]);
    let mut decoder = BpTrellisDecoder::from_dem_str(OVERPRUNING, cfg).unwrap();
    assert!(matches!(
        decoder.decode_outcome(&[]),
        Err(DecoderError::InvalidDimensions { .. })
    ));
    assert!(matches!(
        decoder.decode(&[]),
        Err(DecoderError::InvalidDimensions { .. })
    ));
    let detectors: Vec<u32> = (0..1600).collect();
    let dem = SparseDem {
        mechanisms: vec![
            (5e-324, detectors.clone(), vec![]),
            (5e-324, detectors, vec![0]),
        ],
        num_detectors: 1600,
        num_observables: 1,
        detector_coords: BTreeMap::new(),
    };
    let cfg = BpTrellisConfig {
        k: 1,
        merge_indistinguishable: false,
        escalation: vec![EscalationRung {
            k: 16,
            delta: 100.0,
        }],
        ..BpTrellisConfig::default()
    };
    let mut decoder = BpTrellisDecoder::from_sparse_dem(&dem, cfg).unwrap();
    assert!(matches!(
        decoder.decode_outcome(&vec![0; 1600]),
        Err(DecoderError::InternalError(_))
    ));
    assert!(matches!(
        decoder.decode(&vec![0; 1600]),
        Err(DecoderError::InternalError(_))
    ));
}

#[test]
fn mixed_batch_outcomes_and_strict_results_preserve_order() {
    let dem = format!("error(1) L70\n{OVERPRUNING}\nerror(0.2) D3 D4\ndetector D5");
    let decoder = BpTrellisDecoder::from_dem_str(&dem, config(1, &[(1, 100.0)])).unwrap();
    let shots = vec![
        vec![0; 6],
        vec![0, 0, 0, 0, 0, 1],
        vec![0, 0, 0, 1, 0, 0],
        vec![0, 0, 1, 0, 0, 0],
    ];
    let sequential = decoder.decode_batch_outcomes(&shots, 1).unwrap();
    let parallel = decoder.decode_batch_outcomes(&shots, 4).unwrap();
    assert_eq!(format!("{sequential:?}"), format!("{parallel:?}"));
    let strict_one = decoder.decode_batch(&shots, 1).unwrap();
    let strict_four = decoder.decode_batch(&shots, 4).unwrap();
    assert_eq!(format!("{strict_one:?}"), format!("{strict_four:?}"));
    assert!(matches!(sequential[0], Ok(BpTrellisOutcome::Decoded(_))));
    for (outcome, expected) in sequential.into_iter().skip(1).zip([
        NoPathCause::Residual { detector: 5 },
        NoPathCause::Exhausted,
        NoPathCause::Exhausted,
    ]) {
        assert_eq!(no_path(outcome.unwrap()).cause, expected);
    }
    // Put the parity-locked column first to prove infeasibility before any drops.
    let dem = format!("error(1) L70\nerror(0.2) D3 D4\n{OVERPRUNING}\ndetector D5");
    let decoder = BpTrellisDecoder::from_dem_str(&dem, config(1, &[(1, 100.0)])).unwrap();
    let one = decoder.decode_batch_outcomes(&shots, 1).unwrap();
    let four = decoder.decode_batch_outcomes(&shots, 4).unwrap();
    assert_eq!(format!("{one:?}"), format!("{four:?}"));
    assert_eq!(
        no_path(one[2].as_ref().unwrap().clone()).cause,
        NoPathCause::Infeasible
    );
    assert_eq!(
        format!("{:?}", decoder.decode_batch(&shots, 1).unwrap()),
        format!("{:?}", decoder.decode_batch(&shots, 4).unwrap())
    );
}

#[test]
fn attempt_errors_stay_errors_with_ladder() {
    let mut decoder =
        BpTrellisDecoder::from_dem_str(OVERPRUNING, config(1, &[(16, 100.0)])).unwrap();
    let outcome = decoder.decode_with_attempt(&[0, 0, 0], |inner, _| {
        inner.attempt(PruneParams { k: 0, delta: 100.0 })
    });
    assert!(matches!(
        outcome,
        Err(DecoderError::InvalidConfiguration(_))
    ));
    let outcome = decoder.decode_with_attempt(&[0, 0, 0], |_, _| {
        TrellisDecodeAttempt::Error(DecoderError::InternalError("attempt fault".into()))
    });
    assert!(
        matches!(outcome, Err(DecoderError::InternalError(message)) if message == "attempt fault")
    );
}

#[test]
fn no_path_bp_telemetry_counts_prepare_once() {
    let mut cfg = config(1, &[(1, 100.0)]);
    cfg.bp_score_iterations = 5;
    let mut decoder = BpTrellisDecoder::from_dem_str(OVERPRUNING, cfg).unwrap();
    let before = decoder.inner.bp_refreshes();
    let mut times = Vec::new();
    let report = no_path(
        decoder
            .decode_with_attempt(&[0, 0, 1], |inner, params| {
                let attempt = inner.attempt(params);
                let TrellisDecodeAttempt::NoPath { bp_seconds, .. } = &attempt else {
                    panic!("no-path")
                };
                times.push(bp_seconds.to_bits());
                attempt
            })
            .unwrap(),
    );
    assert_eq!(report.cause, NoPathCause::Exhausted);
    assert_eq!(report.bp_runs, 1);
    assert_eq!(decoder.inner.bp_refreshes() - before, 1);
    assert_eq!(times, vec![report.bp_seconds.to_bits(); 2]);
}

#[test]
fn residual_report_has_no_bp_work_when_bp_enabled() {
    let mut cfg = config(1, &[(2, 100.0), (4, 100.0), (8, 100.0)]);
    cfg.bp_score_iterations = 5;
    let dem = format!("{OVERPRUNING}\ndetector D3");
    let mut decoder = BpTrellisDecoder::from_dem_str(&dem, cfg).unwrap();
    assert!(matches!(
        decoder.decode_outcome(&[0; 4]).unwrap(),
        BpTrellisOutcome::Decoded(_)
    ));
    let before = decoder.inner.bp_refreshes();
    assert_eq!(before, 1);
    let report = counted(&mut decoder, &[0, 0, 0, 1], 0);
    assert_eq!(report.cause, NoPathCause::Residual { detector: 3 });
    assert_eq!(report.bp_runs, 0);
    assert_eq!(report.bp_seconds.to_bits(), 0.0_f64.to_bits());
    assert_eq!(decoder.inner.bp_refreshes(), before);
}
