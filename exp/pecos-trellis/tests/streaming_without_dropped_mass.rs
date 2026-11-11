// Copyright 2026 The PECOS Developers
// SPDX-License-Identifier: Apache-2.0

use pecos_trellis::factor::{Factor, FactorModel, Outcome};
use pecos_trellis::{
    DecoderError, MetricMode, SparseDem, StreamingMode, TrellisConfig, TrellisResult,
    TrellisStreamingDecoder,
};
use std::collections::BTreeMap;

fn retained_bits_equal(full: &TrellisResult, omitted: &TrellisResult) {
    assert!(omitted.dropped_log_mass.is_nan());
    assert_eq!(full.predicted, omitted.predicted);
    assert_eq!(full.log_evidence.to_bits(), omitted.log_evidence.to_bits());
    assert_eq!(
        full.runner_up_gap.map(f64::to_bits),
        omitted.runner_up_gap.map(f64::to_bits)
    );
    assert_eq!(full.peak_retained_states, omitted.peak_retained_states);
    assert_eq!(full.processed_columns, omitted.processed_columns);
    assert_eq!(full.transitions, omitted.transitions);
    assert_eq!(full.dropped_states, omitted.dropped_states);
    assert_eq!(full.bp_seconds.to_bits(), omitted.bp_seconds.to_bits());
    assert_eq!(full.bp_runs, omitted.bp_runs);
    assert_eq!(full.escalation_rungs_used, omitted.escalation_rungs_used);
    assert_eq!(full.status, omitted.status);
    assert_eq!(full.logical_masses.len(), omitted.logical_masses.len());
    for (a, b) in full.logical_masses.iter().zip(&omitted.logical_masses) {
        assert_eq!(a.logical, b.logical);
        assert_eq!(a.log_mass.to_bits(), b.log_mass.to_bits());
    }
}

fn errors_equal(a: &DecoderError, b: &DecoderError) {
    assert_eq!(std::mem::discriminant(a), std::mem::discriminant(b));
    assert_eq!(a.to_string(), b.to_string());
}

fn same_outcome<T: std::fmt::Debug + PartialEq>(
    a: Result<T, DecoderError>,
    b: Result<T, DecoderError>,
) -> bool {
    match (a, b) {
        (Ok(a), Ok(b)) => {
            assert_eq!(a, b);
            true
        }
        (Err(a), Err(b)) => {
            errors_equal(&a, &b);
            false
        }
        values => panic!("full/omitted outcome mismatch: {values:?}"),
    }
}

#[test]
fn callbacks_and_all_retained_terminal_bits_match_both_schedules_and_reset() {
    let mut pruned = false;
    let mut saw_commit = false;
    let mut success_count = 0;
    for width in [12, 65, 131] {
        let row = |d: usize| u32::try_from(d * (width - 1) / 11).unwrap();
        let mut dem = SparseDem {
            mechanisms: Vec::new(),
            detector_coords: BTreeMap::new(),
            num_detectors: width,
            num_observables: 131,
        };
        for d in 0..8 {
            let label = u32::try_from(d).unwrap();
            dem.mechanisms.push((
                0.07 + f64::from(label) * 0.045,
                vec![row(d)],
                vec![label, 64 + label],
            ));
            dem.mechanisms.push((
                0.13 + f64::from(label) * 0.015,
                vec![row(d), row((d + 1) % 8)],
                vec![128 + label % 3],
            ));
        }
        dem.mechanisms.push((0.31, vec![], vec![130]));
        dem.mechanisms.push((0.0, vec![row(9)], vec![129]));
        dem.mechanisms.push((1.0, vec![row(10)], vec![64, 130]));
        for (k, delta, alpha, merge) in [
            (usize::MAX, f64::INFINITY, 0.8, false),
            (1, 1e9, 0.8, false),
            (4, 1.2, 0.8, true),
            (16, 2.0, 0.0, false),
        ] {
            let mut order: Vec<_> = (0..dem.mechanisms.len()).collect();
            if merge {
                order.reverse();
            }
            let config = TrellisConfig {
                k,
                delta,
                score_alpha: alpha,
                merge_indistinguishable: merge,
                column_order: Some(order),
                ..TrellisConfig::default()
            };
            for lag in [None, Some(2), Some(4)] {
                let bounds = vec![0, 0, width / 4, width / 2, width, width];
                let mut full =
                    TrellisStreamingDecoder::from_sparse_dem(&dem, config.clone()).unwrap();
                let mut omitted =
                    TrellisStreamingDecoder::from_sparse_dem(&dem, config.clone()).unwrap();
                if let Some(window_size) = lag {
                    let mode = StreamingMode::FixedLag {
                        detector_round_bounds: bounds.clone(),
                        window_size,
                    };
                    full.set_streaming_mode(mode.clone()).unwrap();
                    omitted.set_streaming_mode(mode).unwrap();
                }
                for shot in 0..64_usize {
                    full.reset();
                    omitted.reset();
                    let mut syndrome = vec![0; width];
                    for bit in 0..8 {
                        syndrome[row(bit) as usize] = if (shot.rotate_left(2) >> bit) & 1 != 0 {
                            7
                        } else {
                            0
                        };
                    }
                    syndrome[row(10) as usize] = 1;
                    assert!(same_outcome(
                        full.advance(),
                        omitted.advance_without_dropped_mass()
                    ));
                    let mut failed = false;
                    for (arrival, pair) in bounds.windows(2).enumerate() {
                        if !same_outcome(
                            full.feed_prefix(&syndrome[pair[0]..pair[1]]),
                            omitted.feed_prefix(&syndrome[pair[0]..pair[1]]),
                        ) {
                            failed = true;
                            break;
                        }
                        // Delay one callback in half the shots, exercising cached-plan fallback.
                        if arrival == 1 && shot % 2 == 0 {
                            continue;
                        }
                        let a = full.advance();
                        let b = omitted.advance_without_dropped_mass();
                        if let Ok(progress) = &a {
                            saw_commit |= !progress.newly_committed.is_empty();
                        }
                        if !same_outcome(a, b) {
                            failed = true;
                            break;
                        }
                        assert_eq!(full.committed(), omitted.committed());
                        assert!(same_outcome(
                            full.advance(),
                            omitted.advance_without_dropped_mass()
                        ));
                    }
                    match (full.flush(), omitted.flush_without_dropped_mass()) {
                        (Ok(a), Ok(b)) => {
                            assert!(!failed);
                            retained_bits_equal(&a, &b);
                            retained_bits_equal(&a, &omitted.flush_without_dropped_mass().unwrap());
                            assert_eq!(full.committed(), omitted.committed());
                            pruned |= a.dropped_states > 0;
                            success_count += 1;
                        }
                        (Err(a), Err(b)) => errors_equal(&a, &b),
                        values => panic!("terminal mismatch: {values:?}"),
                    }
                }
            }
        }
    }
    assert!(pruned);
    assert!(saw_commit);
    assert!(success_count > 500);
}

#[test]
fn mixing_guards_empty_models_and_reset_keep_the_existing_contract() {
    for text in [
        "",
        "error(1) L130",
        "error(0.3) L0",
        "error(0.2) D0 L130\nerror(0.3) D1",
    ] {
        let mut full =
            TrellisStreamingDecoder::from_dem_str(text, TrellisConfig::default()).unwrap();
        let mut omitted =
            TrellisStreamingDecoder::from_dem_str(text, TrellisConfig::default()).unwrap();
        let width = SparseDem::from_dem_str(text).unwrap().num_detectors;
        let syndrome = vec![0; width];
        // A full advance may precede the opt-in call, including already processed columns.
        assert_eq!(full.advance().unwrap(), omitted.advance().unwrap());
        full.feed_dense(&syndrome).unwrap();
        omitted.feed_dense(&syndrome).unwrap();
        assert_eq!(
            full.advance().unwrap(),
            omitted.advance_without_dropped_mass().unwrap()
        );
        let expected = full.flush().unwrap();
        retained_bits_equal(&expected, &omitted.flush_without_dropped_mass().unwrap());
        for error in [
            omitted.advance().unwrap_err(),
            omitted.advance_prediction().unwrap_err(),
            omitted.flush().unwrap_err(),
            omitted.flush_prediction().unwrap_err(),
        ] {
            assert!(matches!(error, DecoderError::InvalidConfiguration(_)));
            assert!(
                error
                    .to_string()
                    .contains("reset before requesting evidence")
            );
        }
        omitted.reset();
        omitted.feed_dense(&syndrome).unwrap();
        let restored = omitted.flush().unwrap();
        assert_eq!(expected, restored);
        assert!(!restored.dropped_log_mass.is_nan());
        // The old prediction-only path keeps its existing guard but can opt into
        // the newly explicit retained terminal result without recovering diagnostics.
        omitted.reset();
        omitted.feed_dense(&syndrome).unwrap();
        omitted.advance_prediction_without_evidence().unwrap();
        retained_bits_equal(&expected, &omitted.flush_without_dropped_mass().unwrap());
        omitted.reset();
        omitted.feed_dense(&syndrome).unwrap();
        omitted.flush_prediction_without_evidence().unwrap();
        assert!(omitted.flush().is_err());
        retained_bits_equal(&expected, &omitted.flush_without_dropped_mass().unwrap());
    }
}

#[test]
fn errors_incomplete_rounds_and_sticky_failures_match() {
    for (text, syndrome) in [
        ("error(0.1) D0 D1 L0", [1, 0]),
        ("error(0.1) D0 L0\ndetector D1", [0, 1]),
        ("error(0.1) D0 L0\nerror(1) D1", [0, 0]),
    ] {
        for fixed in [false, true] {
            let mut a =
                TrellisStreamingDecoder::from_dem_str(text, TrellisConfig::default()).unwrap();
            let mut b =
                TrellisStreamingDecoder::from_dem_str(text, TrellisConfig::default()).unwrap();
            if fixed {
                let mode = StreamingMode::FixedLag {
                    detector_round_bounds: vec![0, 1, 2, 2],
                    window_size: 2,
                };
                a.set_streaming_mode(mode.clone()).unwrap();
                b.set_streaming_mode(mode).unwrap();
            }
            errors_equal(
                &a.flush().unwrap_err(),
                &b.flush_without_dropped_mass().unwrap_err(),
            );
            assert!(same_outcome(
                a.feed_prefix(&syndrome[..1]),
                b.feed_prefix(&syndrome[..1])
            ));
            assert!(same_outcome(a.advance(), b.advance_without_dropped_mass()));
            same_outcome(a.feed_prefix(&syndrome[1..]), b.feed_prefix(&syndrome[1..]));
            same_outcome(a.advance(), b.advance_without_dropped_mass());
            errors_equal(
                &a.flush().unwrap_err(),
                &b.flush_without_dropped_mass().unwrap_err(),
            );
            // Sticky stored failure wins over later incomplete/overflow errors.
            same_outcome(a.feed_prefix(&[0, 0, 0]), b.feed_prefix(&[0, 0, 0]));
            a.reset();
            b.reset();
            let good = if text.contains("error(1)") {
                [0, 1]
            } else {
                [0, 0]
            };
            a.feed_dense(&good).unwrap();
            b.feed_dense(&good).unwrap();
            retained_bits_equal(
                &a.flush().unwrap(),
                &b.flush_without_dropped_mass().unwrap(),
            );
        }
    }
    let mut a = TrellisStreamingDecoder::from_dem_str("error(0.1) D0 L0", TrellisConfig::default())
        .unwrap();
    let mut b = TrellisStreamingDecoder::from_dem_str("error(0.1) D0 L0", TrellisConfig::default())
        .unwrap();
    let mode = StreamingMode::FixedLag {
        detector_round_bounds: vec![0, 1, 1],
        window_size: 2,
    };
    a.set_streaming_mode(mode.clone()).unwrap();
    b.set_streaming_mode(mode).unwrap();
    errors_equal(
        &a.feed_prefix(&[0, 0]).unwrap_err(),
        &b.feed_prefix(&[0, 0]).unwrap_err(),
    );
    a.feed_prefix(&[0]).unwrap();
    b.feed_prefix(&[0]).unwrap();
    errors_equal(
        &a.flush().unwrap_err(),
        &b.flush_without_dropped_mass().unwrap_err(),
    );
    a.feed_prefix(&[]).unwrap();
    b.feed_prefix(&[]).unwrap();
    retained_bits_equal(
        &a.flush().unwrap(),
        &b.flush_without_dropped_mass().unwrap(),
    );
}

#[test]
fn binary_factors_match_and_bp_and_integer_rejections_are_unchanged() {
    let dem = SparseDem::from_dem_str(
        "error(1) D64 L129\nerror(0.2) D65 L65\nerror(0.3) D0 L0\nerror(0.4) D64 L64",
    )
    .unwrap();
    let model = FactorModel::try_from(&dem).unwrap();
    let mut a =
        TrellisStreamingDecoder::from_factor_model(&model, TrellisConfig::default()).unwrap();
    let mut b =
        TrellisStreamingDecoder::from_factor_model(&model, TrellisConfig::default()).unwrap();
    let mut bits = [0; 66];
    bits[0] = 1;
    bits[64] = 1;
    bits[65] = 7;
    for chunk in bits.chunks(17) {
        a.feed_prefix(chunk).unwrap();
        b.feed_prefix(chunk).unwrap();
        assert_eq!(
            a.advance().unwrap(),
            b.advance_without_dropped_mass().unwrap()
        );
    }
    retained_bits_equal(
        &a.flush().unwrap(),
        &b.flush_without_dropped_mass().unwrap(),
    );
    for config in [
        TrellisConfig {
            bp_score_iterations: 1,
            ..TrellisConfig::default()
        },
        TrellisConfig {
            bp_score_iterations: 2,
            k: usize::MAX,
            delta: f64::INFINITY,
            ..TrellisConfig::default()
        },
        TrellisConfig {
            metric_mode: MetricMode::MaxLogInt,
            ..TrellisConfig::default()
        },
    ] {
        let sparse = TrellisStreamingDecoder::from_sparse_dem(&dem, config.clone()).unwrap_err();
        let factor = TrellisStreamingDecoder::from_factor_model(&model, config).unwrap_err();
        errors_equal(&sparse, &factor);
        assert!(matches!(sparse, DecoderError::InvalidConfiguration(_)));
    }
    let nary = FactorModel::new(
        vec![Factor {
            outcomes: vec![
                Outcome {
                    probability: 0.5,
                    detectors: vec![],
                    observables: vec![],
                },
                Outcome {
                    probability: 0.3,
                    detectors: vec![0],
                    observables: vec![],
                },
                Outcome {
                    probability: 0.2,
                    detectors: vec![0],
                    observables: vec![0],
                },
            ],
        }],
        1,
        1,
    )
    .unwrap();
    assert!(
        TrellisStreamingDecoder::from_factor_model(&nary, TrellisConfig::default())
            .unwrap_err()
            .to_string()
            .contains("binary float kernel")
    );
}
