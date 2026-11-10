// Copyright 2026 The PECOS Developers
// SPDX-License-Identifier: Apache-2.0

use pecos_trellis::{
    DecoderError, SparseDem, StreamingMode, TrellisConfig, TrellisResult, TrellisStreamingDecoder,
};
use std::collections::BTreeMap;

fn result_bits_equal(a: &TrellisResult, b: &TrellisResult, omitted: bool) {
    assert_eq!(a.predicted, b.predicted);
    assert_eq!(a.log_evidence.to_bits(), b.log_evidence.to_bits());
    assert_eq!(
        a.runner_up_gap.map(f64::to_bits),
        b.runner_up_gap.map(f64::to_bits)
    );
    assert_eq!(a.peak_retained_states, b.peak_retained_states);
    assert_eq!(a.processed_columns, b.processed_columns);
    assert_eq!(a.transitions, b.transitions);
    assert_eq!(a.dropped_states, b.dropped_states);
    if omitted {
        assert!(b.dropped_log_mass.is_nan());
    } else {
        assert_eq!(a.dropped_log_mass.to_bits(), b.dropped_log_mass.to_bits());
    }
    assert_eq!(a.bp_seconds.to_bits(), b.bp_seconds.to_bits());
    assert_eq!(a.bp_runs, b.bp_runs);
    assert_eq!(a.escalation_rungs_used, b.escalation_rungs_used);
    assert_eq!(a.status, b.status);
    assert_eq!(a.logical_masses.len(), b.logical_masses.len());
    for (a, b) in a.logical_masses.iter().zip(&b.logical_masses) {
        assert_eq!(a.logical, b.logical);
        assert_eq!(a.log_mass.to_bits(), b.log_mass.to_bits());
    }
}

fn config_error(error: DecoderError) {
    assert!(
        matches!(error, DecoderError::InvalidConfiguration(_)),
        "{error}"
    );
}

fn error_equal(a: DecoderError, b: DecoderError) {
    assert_eq!(std::mem::discriminant(&a), std::mem::discriminant(&b));
    assert_eq!(a.to_string(), b.to_string());
}

fn begin(stream: &mut TrellisStreamingDecoder, omitted: bool) -> usize {
    if omitted {
        stream.begin_bounded_advance_without_dropped_mass()
    } else {
        stream.begin_bounded_advance()
    }
    .unwrap()
}

fn drain(stream: &mut TrellisStreamingDecoder, budget: usize) -> usize {
    let commitments = stream.committed();
    let mut remaining = stream.bounded_columns_remaining().unwrap();
    let mut calls = 0;
    while remaining != 0 {
        config_error(stream.finish_bounded_advance().unwrap_err());
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
            for column in 0..140 {
                dem.mechanisms.push((
                    0.01 + (column % 37) as f64 * 0.01,
                    vec![row(round)],
                    vec![(round * 32 + column % 3) as u32],
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
                k,
                delta,
                score_alpha: alpha,
                merge_indistinguishable: merge,
                column_order: Some((0..dem.mechanisms.len()).collect()),
                ..TrellisConfig::default()
            };
            for omitted in [false, true] {
                for budget in [1, 2, 17, 64, 128] {
                    let mut full =
                        TrellisStreamingDecoder::from_sparse_dem(&dem, config.clone()).unwrap();
                    let mut bounded =
                        TrellisStreamingDecoder::from_sparse_dem(&dem, config.clone()).unwrap();
                    for bits in 0..8 {
                        full.reset();
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
                            full.feed_prefix(&syndrome[pair[0]..pair[1]]).unwrap();
                            bounded.feed_prefix(&syndrome[pair[0]..pair[1]]).unwrap();
                            let pending = begin(&mut bounded, omitted);
                            let calls = drain(&mut bounded, budget);
                            saw_many_steps |= calls > 128;
                            if update != 3 {
                                let a = full.advance().unwrap();
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
                            } else {
                                let a = full.flush().unwrap();
                                let b = bounded.finish_bounded_flush().unwrap();
                                result_bits_equal(&a, &b, omitted);
                                assert_eq!(a.processed_columns - last_processed, pending);
                                assert_eq!(full.committed(), bounded.committed());
                                saw_pruning |= a.dropped_states > 0;
                                begin(&mut bounded, omitted);
                                assert_eq!(bounded.bounded_columns_remaining(), Some(0));
                                result_bits_equal(
                                    &a,
                                    &bounded.finish_bounded_flush().unwrap(),
                                    omitted,
                                );
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
    let mut s = TrellisStreamingDecoder::from_dem_str(text, TrellisConfig::default()).unwrap();
    assert_eq!(s.bounded_columns_remaining(), None);
    config_error(s.step_bounded_advance(64).unwrap_err());
    config_error(s.finish_bounded_advance().unwrap_err());
    s.feed_prefix(&[1]).unwrap();
    let target = s.begin_bounded_advance().unwrap();
    assert!(target > 0);
    config_error(s.begin_bounded_advance().unwrap_err());
    config_error(s.begin_bounded_advance_without_dropped_mass().unwrap_err());
    config_error(s.feed_prefix(&[]).unwrap_err());
    config_error(s.feed_prefix(&[0]).unwrap_err());
    config_error(s.feed_dense(&[1, 0]).unwrap_err());
    config_error(s.advance().unwrap_err());
    config_error(s.advance_prediction().unwrap_err());
    config_error(s.advance_without_dropped_mass().unwrap_err());
    config_error(s.advance_prediction_without_evidence().unwrap_err());
    config_error(
        s.set_streaming_mode(StreamingMode::DependencyDriven)
            .unwrap_err(),
    );
    config_error(s.step_bounded_advance(0).unwrap_err());
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
    // Rejected opt-in reentry did not switch the original full drain's mode.
    s.advance().unwrap();
    s.feed_prefix(&[0]).unwrap();
    s.begin_bounded_advance_without_dropped_mass().unwrap();
    drain(&mut s, 1);
    s.finish_bounded_flush().unwrap();
    config_error(s.begin_bounded_advance().unwrap_err());
    config_error(s.advance().unwrap_err());
    s.reset();
    assert_eq!(s.bounded_columns_remaining(), None);
    s.feed_dense(&[1, 0]).unwrap();
    s.flush().unwrap();
    // Reset in the middle of actual work also discards the pending operation.
    s.reset();
    s.feed_dense(&[1, 0]).unwrap();
    s.begin_bounded_advance_without_dropped_mass().unwrap();
    s.step_bounded_advance(1).unwrap();
    s.reset();
    s.feed_dense(&[1, 0]).unwrap();
    assert!(!s.flush().unwrap().dropped_log_mass.is_nan());
}

#[test]
fn unsupported_schedule_and_empty_and_incomplete_models_do_not_change_legacy_paths() {
    for omitted in [false, true] {
        for text in ["", "error(1) L130", "error(0.3) L130"] {
            let mut full =
                TrellisStreamingDecoder::from_dem_str(text, TrellisConfig::default()).unwrap();
            let mut bounded =
                TrellisStreamingDecoder::from_dem_str(text, TrellisConfig::default()).unwrap();
            begin(&mut bounded, omitted);
            drain(&mut bounded, 64);
            result_bits_equal(
                &full.flush().unwrap(),
                &bounded.finish_bounded_flush().unwrap(),
                omitted,
            );
        }
    }
    let mut s = TrellisStreamingDecoder::from_dem_str(
        "error(0.2) D0 L0\nerror(0.3) D1",
        TrellisConfig::default(),
    )
    .unwrap();
    s.set_streaming_mode(StreamingMode::FixedLag {
        detector_round_bounds: vec![0, 1, 2],
        window_size: 2,
    })
    .unwrap();
    config_error(s.begin_bounded_advance().unwrap_err());
    config_error(s.begin_bounded_advance_without_dropped_mass().unwrap_err());
    assert_eq!(s.bounded_columns_remaining(), None);
    s.feed_dense(&[1, 0]).unwrap();
    assert!(!s.flush().unwrap().dropped_log_mass.is_nan());
}

#[test]
fn decoding_failures_remain_sticky_and_reset_restores_both_diagnostic_modes() {
    for (text, syndrome) in [
        ("error(0.1) D0 D1 L0", [1, 0]),
        ("error(0.1) D0 L0\ndetector D1", [0, 1]),
        ("error(0.1) D0 L0\nerror(1) D1", [0, 0]),
    ] {
        for omitted in [false, true] {
            let mut full =
                TrellisStreamingDecoder::from_dem_str(text, TrellisConfig::default()).unwrap();
            let expected = full
                .feed_dense(&syndrome)
                .and_then(|()| full.flush())
                .unwrap_err();
            let mut bounded =
                TrellisStreamingDecoder::from_dem_str(text, TrellisConfig::default()).unwrap();
            let actual = bounded
                .feed_dense(&syndrome)
                .and_then(|()| {
                    begin(&mut bounded, omitted);
                    loop {
                        if bounded.step_bounded_advance(1)?.1 {
                            break;
                        }
                    }
                    bounded.finish_bounded_flush()
                })
                .unwrap_err();
            error_equal(expected, actual);
            let sticky = bounded.step_bounded_advance(1).unwrap_err();
            let message = sticky.to_string();
            for later in [
                bounded.feed_prefix(&[0, 0, 0]).unwrap_err(),
                bounded.begin_bounded_advance().unwrap_err(),
                bounded.finish_bounded_advance().unwrap_err(),
                bounded.finish_bounded_flush().unwrap_err(),
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
            begin(&mut bounded, omitted);
            drain(&mut bounded, 2);
            bounded.finish_bounded_flush().unwrap();
            bounded.reset();
            bounded.feed_dense(&valid).unwrap();
            bounded.flush().unwrap();
        }
    }
}
