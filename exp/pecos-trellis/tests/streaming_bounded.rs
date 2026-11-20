// Copyright 2026 The PECOS Developers
// SPDX-License-Identifier: Apache-2.0

use pecos_trellis::{
    DecoderError, SparseDem, StreamingMode, TrellisConfig, TrellisResult, TrellisStreamingDecoder,
};
use std::collections::BTreeMap;

fn result_bits_equal(a: &TrellisResult, b: &TrellisResult, flag_off: bool) {
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
    } = a;
    assert_eq!(*predicted, b.predicted);
    assert_eq!(log_evidence.to_bits(), b.log_evidence.to_bits());
    assert_eq!(
        runner_up_gap.map(f64::to_bits),
        b.runner_up_gap.map(f64::to_bits)
    );
    assert_eq!(*peak_retained_states, b.peak_retained_states);
    assert_eq!(*processed_columns, b.processed_columns);
    assert_eq!(*transitions, b.transitions);
    assert_eq!(*dropped_states, b.dropped_states);
    if flag_off {
        assert_eq!(b.dropped_log_mass, None);
        assert!(dropped_log_mass.is_some());
    } else {
        assert_eq!(
            dropped_log_mass.expect("enabled diagnostic").to_bits(),
            b.dropped_log_mass.expect("enabled diagnostic").to_bits()
        );
    }
    assert_eq!(bp_seconds.to_bits(), b.bp_seconds.to_bits());
    assert_eq!(*bp_runs, b.bp_runs);
    assert_eq!(*escalation_rungs_used, b.escalation_rungs_used);
    assert_eq!(*status, b.status);
    assert_eq!(logical_masses.len(), b.logical_masses.len());
    for (a, b) in logical_masses.iter().zip(&b.logical_masses) {
        let pecos_trellis::TrellisLogicalMass { logical, log_mass } = a;
        assert_eq!(*logical, b.logical);
        assert_eq!(log_mass.to_bits(), b.log_mass.to_bits());
    }
}

fn config_error(error: &DecoderError) {
    assert!(
        matches!(error, DecoderError::InvalidConfiguration(_)),
        "{error}"
    );
}

fn error_equal(a: &DecoderError, b: &DecoderError) {
    assert_eq!(std::mem::discriminant(a), std::mem::discriminant(b));
    assert_eq!(a.to_string(), b.to_string());
}

fn begin(stream: &mut TrellisStreamingDecoder) -> usize {
    stream.begin_bounded_advance().unwrap()
}

fn drain(stream: &mut TrellisStreamingDecoder, budget: usize) -> usize {
    let commitments = stream.committed();
    let mut remaining = stream.bounded_columns_remaining().unwrap();
    let mut calls = 0;
    while remaining != 0 {
        config_error(&stream.finish_bounded_advance().unwrap_err());
        let (processed, done) = stream.step_bounded_advance(budget).unwrap();
        let after = stream.bounded_columns_remaining().unwrap();
        assert_eq!(remaining - after, budget.min(remaining));
        assert_eq!(done, after == 0);
        assert_eq!(
            stream.committed(),
            commitments,
            "no mid-callback commitment scan"
        );
        assert!(processed > 0);
        remaining = after;
        calls += 1;
    }
    // Zero-work steps do not consume the pending completion or change any state.
    let result = stream.step_bounded_advance(budget).unwrap();
    assert!(result.1);
    assert_eq!(stream.step_bounded_advance(usize::MAX).unwrap(), result);
    calls
}

#[test]
fn every_budget_preserves_every_callback_and_all_terminal_mass_bits() {
    let mut saw_pruning = false;
    let mut saw_many_steps = false;
    let mut saw_early_commit = false;
    for width in [4, 67, 131] {
        let row = |d: usize| u32::try_from(d * (width - 1) / 3).unwrap();
        let mut dem = SparseDem {
            mechanisms: Vec::new(),
            detector_coords: BTreeMap::new(),
            num_detectors: width,
            num_observables: 131,
        };
        // Each update releases >128 columns. Different global logical bits,
        // duplicate routes and detector closure exercise merge/prune folding.
        for round in 0..4 {
            let round_label = u32::try_from(round).unwrap();
            for column in 0..140 {
                let column_label = u32::try_from(column).unwrap();
                dem.mechanisms.push((
                    0.01 + f64::from(column_label % 37) * 0.01,
                    vec![row(round)],
                    vec![round_label * 32 + column_label % 3],
                ));
            }
        }
        dem.mechanisms.push((0.0, vec![row(2)], vec![129]));
        dem.mechanisms.push((1.0, vec![], vec![130]));
        for (k, delta, alpha, merge) in [
            (1, 1e9, 0.8, false),
            (4, 1.1, 0.8, true),
            (16, 2.0, 0.0, false),
        ] {
            let config = TrellisConfig {
                dropped_mass: true,
                k,
                delta,
                score_alpha: alpha,
                merge_indistinguishable: merge,
                column_order: Some((0..dem.mechanisms.len()).collect()),
                ..TrellisConfig::default()
            };
            for flag_off in [false, true] {
                for budget in [1, 2, 17, 64, 128] {
                    let mut flag_on =
                        TrellisStreamingDecoder::from_sparse_dem(&dem, config.clone()).unwrap();
                    let mut bounded = TrellisStreamingDecoder::from_sparse_dem(
                        &dem,
                        TrellisConfig {
                            dropped_mass: !flag_off,
                            ..config.clone()
                        },
                    )
                    .unwrap();
                    for bits in 0..8 {
                        flag_on.reset();
                        bounded.reset();
                        let mut syndrome = vec![0; width];
                        for round in 0..4 {
                            syndrome[row(round) as usize] =
                                if bits >> round & 1 == 0 { 0 } else { 7 };
                        }
                        let bounds = [
                            0,
                            row(0) as usize + 1,
                            row(1) as usize + 1,
                            row(2) as usize + 1,
                            width,
                        ];
                        let mut last_processed = 0;
                        for (update, pair) in bounds.windows(2).enumerate() {
                            flag_on.feed_prefix(&syndrome[pair[0]..pair[1]]).unwrap();
                            bounded.feed_prefix(&syndrome[pair[0]..pair[1]]).unwrap();
                            let pending = begin(&mut bounded);
                            let calls = drain(&mut bounded, budget);
                            saw_many_steps |= calls > 128;
                            if update == 3 {
                                let a = flag_on.flush().unwrap();
                                let b = bounded.finish_bounded_flush().unwrap();
                                result_bits_equal(&a, &b, flag_off);
                                assert_eq!(a.processed_columns - last_processed, pending);
                                assert_eq!(flag_on.committed(), bounded.committed());
                                saw_pruning |= a.dropped_states > 0;
                                begin(&mut bounded);
                                assert_eq!(bounded.bounded_columns_remaining(), Some(0));
                                result_bits_equal(
                                    &a,
                                    &bounded.finish_bounded_flush().unwrap(),
                                    flag_off,
                                );
                            } else {
                                let a = flag_on.advance().unwrap();
                                let b = bounded.finish_bounded_advance().unwrap();
                                assert_eq!(
                                    a, b,
                                    "width={width} k={k} budget={budget} update={update}"
                                );
                                assert_eq!(a.columns_processed - last_processed, pending);
                                saw_early_commit |=
                                    a.newly_committed.iter().any(|&(bit, _)| bit < 100);
                                last_processed = a.columns_processed;
                                assert_eq!(bounded.bounded_columns_remaining(), None);
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(saw_pruning);
    assert!(saw_many_steps);
    assert!(saw_early_commit);
}

#[test]
fn pending_guards_freeze_inputs_modes_and_commitment_cadence_until_finish() {
    let text = "error(0.2) D0 L0\nerror(0.3) D0 L1\nerror(0.4) D1 L130";
    let mut s = TrellisStreamingDecoder::from_dem_str(
        text,
        TrellisConfig {
            dropped_mass: true,
            ..TrellisConfig::default()
        },
    )
    .unwrap();
    assert_eq!(s.bounded_columns_remaining(), None);
    config_error(&s.step_bounded_advance(64).unwrap_err());
    config_error(&s.finish_bounded_advance().unwrap_err());
    s.feed_prefix(&[1]).unwrap();
    let target = s.begin_bounded_advance().unwrap();
    assert!(target > 0);
    config_error(&s.begin_bounded_advance().unwrap_err());
    config_error(&s.feed_prefix(&[]).unwrap_err());
    config_error(&s.feed_prefix(&[0]).unwrap_err());
    config_error(&s.feed_dense(&[1, 0]).unwrap_err());
    config_error(&s.advance().unwrap_err());
    config_error(&s.advance_prediction().unwrap_err());
    config_error(
        &s.set_streaming_mode(StreamingMode::DependencyDriven)
            .unwrap_err(),
    );
    config_error(&s.step_bounded_advance(0).unwrap_err());
    assert_eq!(s.bounded_columns_remaining(), Some(target));
    drain(&mut s, 1);
    assert!(matches!(
        s.finish_bounded_flush(),
        Err(DecoderError::InvalidDimensions {
            expected: 2,
            actual: 1
        })
    ));
    assert_eq!(s.bounded_columns_remaining(), Some(0));
    s.finish_bounded_advance().unwrap();
    // Rejected reentry leaves the original drain intact.
    s.advance().unwrap();
    s.feed_prefix(&[0]).unwrap();
    s.begin_bounded_advance().unwrap();
    drain(&mut s, 1);
    s.finish_bounded_flush().unwrap();
    s.advance().unwrap();
    s.reset();
    assert_eq!(s.bounded_columns_remaining(), None);
    s.feed_dense(&[1, 0]).unwrap();
    s.flush().unwrap();
    // Reset in the middle of actual work also discards the pending operation.
    s.reset();
    s.feed_dense(&[1, 0]).unwrap();
    s.begin_bounded_advance().unwrap();
    s.step_bounded_advance(1).unwrap();
    s.reset();
    s.feed_dense(&[1, 0]).unwrap();
    assert!(s.flush().unwrap().dropped_log_mass.is_some());
}

#[test]
fn unsupported_schedule_and_empty_and_incomplete_models_preserve_configured_diagnostics() {
    for flag_off in [false, true] {
        for text in ["", "error(1) L130", "error(0.3) L130"] {
            let mut flag_on = TrellisStreamingDecoder::from_dem_str(
                text,
                TrellisConfig {
                    dropped_mass: true,
                    ..TrellisConfig::default()
                },
            )
            .unwrap();
            let mut bounded = TrellisStreamingDecoder::from_dem_str(
                text,
                TrellisConfig {
                    dropped_mass: !flag_off,
                    ..TrellisConfig::default()
                },
            )
            .unwrap();
            begin(&mut bounded);
            drain(&mut bounded, 64);
            result_bits_equal(
                &flag_on.flush().unwrap(),
                &bounded.finish_bounded_flush().unwrap(),
                flag_off,
            );
        }
    }
    let mut s = TrellisStreamingDecoder::from_dem_str(
        "error(0.2) D0 L0\nerror(0.3) D1",
        TrellisConfig {
            dropped_mass: true,
            ..TrellisConfig::default()
        },
    )
    .unwrap();
    s.set_streaming_mode(StreamingMode::FixedLag {
        detector_round_bounds: vec![0, 1, 2],
        window_size: 2,
    })
    .unwrap();
    config_error(&s.begin_bounded_advance().unwrap_err());
    assert_eq!(s.bounded_columns_remaining(), None);
    s.feed_dense(&[1, 0]).unwrap();
    assert!(s.flush().unwrap().dropped_log_mass.is_some());
}

#[test]
fn decoding_failures_remain_sticky_and_reset_preserves_each_flag_value() {
    for (text, syndrome) in [
        ("error(0.1) D0 D1 L0", [1, 0]),
        ("error(0.1) D0 L0\ndetector D1", [0, 1]),
        ("error(0.1) D0 L0\nerror(1) D1", [0, 0]),
    ] {
        let text = format!("{text}\nerror(0.25) L1\nerror(0.2) L2");
        for flag_off in [false, true] {
            let mut flag_on = TrellisStreamingDecoder::from_dem_str(
                &text,
                TrellisConfig {
                    k: 1,
                    dropped_mass: true,
                    ..TrellisConfig::default()
                },
            )
            .unwrap();
            let expected = flag_on
                .feed_dense(&syndrome)
                .and_then(|()| flag_on.flush())
                .unwrap_err();
            let mut bounded = TrellisStreamingDecoder::from_dem_str(
                &text,
                TrellisConfig {
                    k: 1,
                    dropped_mass: !flag_off,
                    ..TrellisConfig::default()
                },
            )
            .unwrap();
            let actual = bounded
                .feed_dense(&syndrome)
                .and_then(|()| {
                    begin(&mut bounded);
                    loop {
                        if bounded.step_bounded_advance(1)?.1 {
                            break;
                        }
                    }
                    bounded.finish_bounded_flush()
                })
                .unwrap_err();
            error_equal(&expected, &actual);
            let sticky = bounded.step_bounded_advance(1).unwrap_err();
            let message = sticky.to_string();
            // A failed step leaves its drain pending; every API, including the
            // unbounded and prediction-only ones, must still report the failure.
            for later in [
                bounded.feed_prefix(&[0, 0, 0]).unwrap_err(),
                bounded.begin_bounded_advance().unwrap_err(),
                bounded.finish_bounded_advance().unwrap_err(),
                bounded.finish_bounded_flush().unwrap_err(),
                bounded.advance().unwrap_err(),
                bounded.advance_prediction().unwrap_err(),
                bounded.flush().unwrap_err(),
                bounded.flush_prediction().unwrap_err(),
            ] {
                assert_eq!(later.to_string(), message);
            }
            bounded.reset();
            let valid = if text.contains("error(1)") {
                [0, 1]
            } else {
                [0, 0]
            };
            bounded.feed_dense(&valid).unwrap();
            begin(&mut bounded);
            drain(&mut bounded, 2);
            let result = bounded.finish_bounded_flush().unwrap();
            assert!(result.dropped_states > 0);
            assert_eq!(result.dropped_log_mass.is_some(), !flag_off);
            if let Some(mass) = result.dropped_log_mass {
                assert!(mass.is_finite());
            }
            bounded.reset();
            bounded.feed_dense(&valid).unwrap();
            let result = bounded.flush().unwrap();
            assert!(result.dropped_states > 0);
            assert_eq!(result.dropped_log_mass.is_some(), !flag_off);
            if let Some(mass) = result.dropped_log_mass {
                assert!(mass.is_finite());
            }
        }
    }
}
