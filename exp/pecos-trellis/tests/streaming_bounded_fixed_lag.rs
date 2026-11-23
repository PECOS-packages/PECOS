// Copyright 2026 The PECOS Developers
// SPDX-License-Identifier: Apache-2.0

use pecos_trellis::{
    DecoderError, SparseDem, StreamingMode, TrellisConfig, TrellisResult, TrellisStreamingDecoder,
};
use std::collections::BTreeMap;

fn result_bits_equal(a: &TrellisResult, b: &TrellisResult) {
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
    assert_eq!(
        dropped_log_mass.map(f64::to_bits),
        b.dropped_log_mass.map(f64::to_bits)
    );
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

fn make_stream(
    dem: &SparseDem,
    config: &TrellisConfig,
    bounds: &[usize],
    window_size: usize,
) -> TrellisStreamingDecoder {
    let mut stream = TrellisStreamingDecoder::from_sparse_dem(dem, config.clone()).unwrap();
    stream
        .set_streaming_mode(StreamingMode::FixedLag {
            detector_round_bounds: bounds.to_vec(),
            window_size,
        })
        .unwrap();
    stream
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
        assert!(processed >= remaining - after);
        assert_eq!(stream.committed(), commitments, "no yield-time commitments");
        remaining = after;
        calls += 1;
    }
    // A yielded or already drained step never starts a new release phase.
    let idle = stream.step_bounded_advance(budget).unwrap();
    assert!(idle.1);
    assert_eq!(stream.step_bounded_advance(usize::MAX).unwrap(), idle);
    assert_eq!(stream.committed(), commitments);
    calls
}

fn model(block_width: usize) -> SparseDem {
    let mut mechanisms = Vec::new();
    for round in 0_u32..12 {
        for branch in 0_u32..12 {
            let row = round * u32::try_from(block_width).unwrap();
            let mut detectors = vec![row];
            if round < 11 && branch % 3 == 0 {
                detectors.push(row + u32::try_from(block_width).unwrap());
            }
            if round < 10 && branch % 5 == 0 {
                detectors.push(row + 2 * u32::try_from(block_width).unwrap());
            }
            mechanisms.push((
                0.02 + f64::from(branch % 7) * 0.025,
                detectors,
                vec![(round * 13 + branch % 6) % 131],
            ));
        }
    }
    SparseDem {
        mechanisms,
        detector_coords: BTreeMap::new(),
        num_detectors: 12 * block_width,
        num_observables: 131,
    }
}

#[test]
fn every_budget_matches_unbounded_fixed_lag_at_the_same_arrival_cadence() {
    let mut saw_pruning = false;
    let mut saw_many_steps = false;
    let mut saw_zero_work = false;
    let mut saw_terminal_tail = false;
    for block_width in [1, 17] {
        let dem = model(block_width);
        for empty_rounds in [false, true] {
            let mut bounds: Vec<_> = (0..=12).map(|r| r * block_width).collect();
            if empty_rounds {
                bounds.insert(0, 0);
                bounds.push(dem.num_detectors);
            }
            for (k, delta, alpha, merge) in [(8, 1e9, 0.8, false), (4, 1.3, 0.0, true)] {
                for dropped_mass in [false, true] {
                    let config = TrellisConfig {
                        k,
                        delta,
                        score_alpha: alpha,
                        merge_indistinguishable: merge,
                        column_order: Some((0..dem.mechanisms.len()).collect()),
                        dropped_mass,
                        ..TrellisConfig::default()
                    };
                    for window_size in [2, 3, 8] {
                        // Cadence two intentionally skips a release phase. Its
                        // visible prefix must match unbounded cadence two, not
                        // an oracle that advances after every detector block.
                        for cadence in [1, 2] {
                            for budget in [1, 2, 7, 32, 64, usize::MAX] {
                                let mut full = make_stream(&dem, &config, &bounds, window_size);
                                let mut bounded = make_stream(&dem, &config, &bounds, window_size);
                                let syndrome = vec![0; dem.num_detectors];
                                let mut last_processed = 0;
                                for (round, pair) in bounds.windows(2).enumerate() {
                                    full.feed_prefix(&syndrome[pair[0]..pair[1]]).unwrap();
                                    bounded.feed_prefix(&syndrome[pair[0]..pair[1]]).unwrap();
                                    let terminal = round + 2 == bounds.len();
                                    if !terminal && (round + 1) % cadence != 0 {
                                        continue;
                                    }
                                    let pending = bounded.begin_bounded_advance().unwrap();
                                    saw_zero_work |= pending == 0;
                                    saw_terminal_tail |= terminal && pending > 32;
                                    let calls = drain(&mut bounded, budget);
                                    saw_many_steps |= calls > 32;
                                    if terminal {
                                        let a = full.flush().unwrap();
                                        let b = bounded.finish_bounded_flush().unwrap();
                                        result_bits_equal(&a, &b);
                                        assert_eq!(a.processed_columns - last_processed, pending);
                                        assert_eq!(full.committed(), bounded.committed());
                                        assert_eq!(b.dropped_log_mass.is_some(), dropped_mass);
                                        assert!(!b.logical_masses.is_empty());
                                        saw_pruning |= b.dropped_states > 0;
                                        assert_eq!(bounded.begin_bounded_advance().unwrap(), 0);
                                        drain(&mut bounded, budget);
                                        result_bits_equal(
                                            &a,
                                            &bounded.finish_bounded_flush().unwrap(),
                                        );
                                    } else {
                                        let a = full.advance().unwrap();
                                        let b = bounded.finish_bounded_advance().unwrap();
                                        assert_eq!(a, b);
                                        assert_eq!(a.columns_processed - last_processed, pending);
                                        last_processed = a.columns_processed;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert!(saw_pruning);
    assert!(saw_many_steps);
    assert!(saw_zero_work);
    assert!(saw_terminal_tail);
}

#[test]
fn nonzero_syndromes_and_wide_masks_preserve_stride_two_callback_results() {
    let mut dem = model(17);
    // Independent detector rows guarantee nonzero inputs are reachable even
    // after finite-width pruning; the matrix above exercises cross-round edges.
    for (_, detectors, _) in &mut dem.mechanisms {
        detectors.truncate(1);
    }
    let bounds: Vec<_> = (0..=12).map(|round| round * 17).collect();
    let mut syndrome = vec![0; dem.num_detectors];
    for round in 0..12 {
        syndrome[round * 17] = u8::from(round % 3 != 0);
    }
    for dropped_mass in [false, true] {
        let config = TrellisConfig {
            k: 8,
            delta: 1e9,
            score_alpha: 0.8,
            column_order: Some((0..dem.mechanisms.len()).collect()),
            dropped_mass,
            ..TrellisConfig::default()
        };
        for budget in [1, 32] {
            let mut full = make_stream(&dem, &config, &bounds, 8);
            let mut bounded = make_stream(&dem, &config, &bounds, 8);
            for (round, pair) in bounds.windows(2).enumerate() {
                full.feed_prefix(&syndrome[pair[0]..pair[1]]).unwrap();
                bounded.feed_prefix(&syndrome[pair[0]..pair[1]]).unwrap();
                if round % 2 == 0 {
                    continue;
                }
                bounded.begin_bounded_advance().unwrap();
                drain(&mut bounded, budget);
                if round == 11 {
                    result_bits_equal(
                        &full.flush().unwrap(),
                        &bounded.finish_bounded_flush().unwrap(),
                    );
                    assert_eq!(full.committed(), bounded.committed());
                } else {
                    assert_eq!(
                        full.advance().unwrap(),
                        bounded.finish_bounded_advance().unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn fixed_lag_pending_guards_freeze_inputs_schedule_and_terminal_phase() {
    let dem = model(1);
    let bounds: Vec<_> = (0..=12).collect();
    let config = TrellisConfig::default();
    let mut stream = make_stream(&dem, &config, &bounds, 8);
    // No input means no fixed-lag column work, but the explicit drain must
    // still finish before the caller can feed a detector block.
    assert_eq!(stream.begin_bounded_advance().unwrap(), 0);
    config_error(&stream.feed_prefix(&[0]).unwrap_err());
    assert!(matches!(
        stream.finish_bounded_flush(),
        Err(DecoderError::InvalidDimensions {
            expected: 12,
            actual: 0
        })
    ));
    drain(&mut stream, 32);
    assert_eq!(
        stream.finish_bounded_advance().unwrap().columns_processed,
        0
    );
    for _ in 0..8 {
        stream.feed_prefix(&[0]).unwrap();
    }
    let pending = stream.begin_bounded_advance().unwrap();
    assert!(pending > 1);
    config_error(&stream.begin_bounded_advance().unwrap_err());
    config_error(&stream.step_bounded_advance(0).unwrap_err());
    config_error(&stream.feed_prefix(&[0]).unwrap_err());
    config_error(&stream.feed_dense(&[0; 12]).unwrap_err());
    config_error(&stream.advance().unwrap_err());
    config_error(&stream.advance_prediction().unwrap_err());
    assert!(matches!(
        stream.flush(),
        Err(DecoderError::InvalidDimensions {
            expected: 12,
            actual: 8
        })
    ));
    config_error(
        &stream
            .set_streaming_mode(StreamingMode::DependencyDriven)
            .unwrap_err(),
    );
    assert_eq!(stream.bounded_columns_remaining(), Some(pending));
    stream.step_bounded_advance(1).unwrap();
    config_error(&stream.finish_bounded_advance().unwrap_err());
    drain(&mut stream, 2);
    assert!(matches!(
        stream.finish_bounded_flush(),
        Err(DecoderError::InvalidDimensions {
            expected: 12,
            actual: 8
        })
    ));
    stream.finish_bounded_advance().unwrap();
    // Reset also cancels a partly drained operation, retaining the schedule.
    stream.feed_prefix(&[0]).unwrap();
    stream.begin_bounded_advance().unwrap();
    stream.step_bounded_advance(1).unwrap();
    stream.reset();
    assert_eq!(stream.bounded_columns_remaining(), None);
    stream.feed_dense(&[0; 12]).unwrap();
    let all_columns = stream.begin_bounded_advance().unwrap();
    assert!(all_columns > 32);
    stream.step_bounded_advance(32).unwrap();
    // Terminal construction cannot silently run the uncaptured tail.
    config_error(&stream.finish_bounded_flush().unwrap_err());
    assert_eq!(stream.bounded_columns_remaining(), Some(all_columns - 32));
    drain(&mut stream, 32);
    stream.finish_bounded_flush().unwrap();
}

#[test]
fn terminal_empty_round_is_required_and_its_tail_is_bounded() {
    let dem = model(1);
    let bounds = [0, 6, 12, 12];
    for dropped_mass in [false, true] {
        let config = TrellisConfig {
            dropped_mass,
            ..TrellisConfig::default()
        };
        let mut full = make_stream(&dem, &config, &bounds, 2);
        let mut bounded = make_stream(&dem, &config, &bounds, 2);
        for _ in 0..2 {
            full.feed_prefix(&[0; 6]).unwrap();
            bounded.feed_prefix(&[0; 6]).unwrap();
            bounded.begin_bounded_advance().unwrap();
            drain(&mut bounded, 7);
            assert_eq!(
                full.advance().unwrap(),
                bounded.finish_bounded_advance().unwrap()
            );
        }
        bounded.begin_bounded_advance().unwrap();
        drain(&mut bounded, 7);
        config_error(&full.flush().unwrap_err());
        config_error(&bounded.finish_bounded_flush().unwrap_err());
        bounded.finish_bounded_advance().unwrap();
        full.feed_prefix(&[]).unwrap();
        bounded.feed_prefix(&[]).unwrap();
        assert!(bounded.begin_bounded_advance().unwrap() > 7);
        config_error(&bounded.finish_bounded_flush().unwrap_err());
        drain(&mut bounded, 7);
        result_bits_equal(
            &full.flush().unwrap(),
            &bounded.finish_bounded_flush().unwrap(),
        );
    }
}

#[test]
fn inconsistent_syndrome_failures_stay_sticky_across_bounded_fixed_lag_calls() {
    for (text, syndrome) in [
        ("error(0.1) D0 D1 L0", [1, 0]),
        ("error(0.1) D0 L0\ndetector D1", [0, 1]),
    ] {
        for dropped_mass in [false, true] {
            let config = TrellisConfig {
                dropped_mass,
                ..TrellisConfig::default()
            };
            let mut full = TrellisStreamingDecoder::from_dem_str(text, config.clone()).unwrap();
            let mut bounded = TrellisStreamingDecoder::from_dem_str(text, config).unwrap();
            let mode = StreamingMode::FixedLag {
                detector_round_bounds: vec![0, 1, 2],
                window_size: 2,
            };
            full.set_streaming_mode(mode.clone()).unwrap();
            bounded.set_streaming_mode(mode).unwrap();
            full.feed_prefix(&syndrome[..1]).unwrap();
            bounded.feed_prefix(&syndrome[..1]).unwrap();
            assert_eq!(bounded.begin_bounded_advance().unwrap(), 0);
            drain(&mut bounded, 1);
            assert_eq!(
                full.advance().unwrap(),
                bounded.finish_bounded_advance().unwrap()
            );
            let expected = full
                .feed_prefix(&syndrome[1..])
                .and_then(|()| full.flush())
                .unwrap_err();
            let actual = bounded
                .feed_prefix(&syndrome[1..])
                .and_then(|()| {
                    bounded.begin_bounded_advance()?;
                    loop {
                        if bounded.step_bounded_advance(1)?.1 {
                            break;
                        }
                    }
                    bounded.finish_bounded_flush()
                })
                .unwrap_err();
            error_equal(&expected, &actual);
            error_equal(&expected, &bounded.begin_bounded_advance().unwrap_err());
            error_equal(&expected, &bounded.step_bounded_advance(0).unwrap_err());
            error_equal(&expected, &bounded.finish_bounded_advance().unwrap_err());
            error_equal(&expected, &bounded.finish_bounded_flush().unwrap_err());
            error_equal(&expected, &bounded.feed_prefix(&[]).unwrap_err());
            bounded.reset();
            bounded.feed_dense(&[0, 0]).unwrap();
            bounded.begin_bounded_advance().unwrap();
            drain(&mut bounded, 1);
            assert_eq!(
                bounded
                    .finish_bounded_flush()
                    .unwrap()
                    .dropped_log_mass
                    .is_some(),
                dropped_mass
            );
        }
    }
}

#[test]
fn detector_free_models_wait_for_explicit_empty_fixed_lag_arrivals() {
    for text in ["", "error(1) L130", "error(0.3) L130"] {
        for dropped_mass in [false, true] {
            let config = TrellisConfig {
                dropped_mass,
                ..TrellisConfig::default()
            };
            let mut full = TrellisStreamingDecoder::from_dem_str(text, config.clone()).unwrap();
            let mut bounded = TrellisStreamingDecoder::from_dem_str(text, config).unwrap();
            let mode = StreamingMode::FixedLag {
                detector_round_bounds: vec![0, 0, 0, 0],
                window_size: 2,
            };
            full.set_streaming_mode(mode.clone()).unwrap();
            bounded.set_streaming_mode(mode).unwrap();
            assert_eq!(bounded.begin_bounded_advance().unwrap(), 0);
            drain(&mut bounded, 1);
            config_error(&bounded.finish_bounded_flush().unwrap_err());
            assert_eq!(
                full.advance().unwrap(),
                bounded.finish_bounded_advance().unwrap()
            );
            for round in 0..3 {
                full.feed_prefix(&[]).unwrap();
                bounded.feed_prefix(&[]).unwrap();
                bounded.begin_bounded_advance().unwrap();
                drain(&mut bounded, 1);
                if round == 2 {
                    result_bits_equal(
                        &full.flush().unwrap(),
                        &bounded.finish_bounded_flush().unwrap(),
                    );
                } else {
                    assert_eq!(
                        full.advance().unwrap(),
                        bounded.finish_bounded_advance().unwrap()
                    );
                }
            }
        }
    }
}
