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

use pecos_trellis::factor::{Factor, FactorModel, Outcome};
use pecos_trellis::{
    DecoderError, MetricMode, SparseDem, TrellisConfig, TrellisDecoder, TrellisResult,
    TrellisStreamingDecoder,
};
use rand::seq::SliceRandom;
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use std::collections::BTreeMap;

fn assert_bit_identical(left: &TrellisResult, right: &TrellisResult) {
    assert_eq!(left.predicted, right.predicted);
    assert_eq!(left.log_evidence.to_bits(), right.log_evidence.to_bits());
    assert_eq!(
        left.runner_up_gap.map(f64::to_bits),
        right.runner_up_gap.map(f64::to_bits)
    );
    assert_eq!(left.peak_retained_states, right.peak_retained_states);
    assert_eq!(left.processed_columns, right.processed_columns);
    assert_eq!(left.transitions, right.transitions);
    assert_eq!(left.dropped_states, right.dropped_states);
    assert_eq!(
        left.dropped_log_mass.to_bits(),
        right.dropped_log_mass.to_bits()
    );
    assert_eq!(left.bp_seconds.to_bits(), right.bp_seconds.to_bits());
    assert_eq!(left.escalation_rungs_used, right.escalation_rungs_used);
    assert_eq!(left.status, right.status);
    assert_eq!(left.logical_masses.len(), right.logical_masses.len());
    for (left, right) in left.logical_masses.iter().zip(&right.logical_masses) {
        assert_eq!(left.logical, right.logical);
        assert_eq!(left.log_mass.to_bits(), right.log_mass.to_bits());
    }
}

fn assert_no_path(error: &DecoderError, expected: &DecoderError) {
    assert!(matches!(error, DecoderError::DecodingFailed(_)));
    assert!(matches!(expected, DecoderError::DecodingFailed(_)));
    assert_eq!(error.to_string(), expected.to_string());
}

#[test]
fn random_rounds_match_batch_for_every_chunk_size_and_reset() {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xdec0_de42);
    let mut saw_early_commitment = false;
    for case in 0..24 {
        let width = [12, 65, 129][case % 3];
        let mut dem = SparseDem {
            mechanisms: Vec::new(),
            detector_coords: BTreeMap::new(),
            num_detectors: width,
            num_observables: 70,
        };
        for round in 0..4_u32 {
            for local in 0..3_u32 {
                let logicals = if local == 0 {
                    vec![round, 65 + round]
                } else {
                    vec![]
                };
                dem.mechanisms.push((
                    rng.random_range(0.02..0.48),
                    vec![round * 3 + local],
                    logicals,
                ));
            }
            if case != 0 {
                dem.mechanisms.push((
                    rng.random_range(0.02..0.48),
                    vec![round * 3, round * 3 + 1],
                    vec![round],
                ));
            }
        }
        if case % 3 == 1 {
            // Stable time order with mechanisms coupling neighboring rounds.
            for round in 0..3_u32 {
                dem.mechanisms.push((
                    rng.random_range(0.02..0.48),
                    vec![round * 3 + 2, round * 3 + 3],
                    vec![round],
                ));
            }
            dem.mechanisms.sort_by_key(|(_, detectors, _)| detectors[0]);
        }
        // Exercise probability-zero filtering, forced logicals, and early active
        // future detectors from forced syndrome folding.
        dem.mechanisms.push((0.0, vec![11], vec![0]));
        dem.mechanisms
            .push((1.0, if case % 2 == 0 { vec![10] } else { vec![] }, vec![64]));
        // Spread the same coupled rounds across detector word boundaries.
        for (_, detectors, _) in &mut dem.mechanisms {
            for detector in detectors {
                *detector = u32::try_from((*detector as usize) * (width - 1) / 11).unwrap();
            }
        }
        let mut order: Vec<_> = (0..dem.mechanisms.len()).collect();
        if case != 0 {
            order.shuffle(&mut rng);
        }
        let config = TrellisConfig {
            column_order: Some(order),
            k: [usize::MAX, 1, 4, 16][case % 4],
            delta: if case % 3 == 0 { f64::INFINITY } else { 2.0 },
            score_alpha: if case % 5 == 0 { 0.0 } else { 0.8 },
            merge_indistinguishable: case % 2 == 0,
            ..TrellisConfig::default()
        };
        let mut batch = TrellisDecoder::from_sparse_dem(&dem, config.clone()).unwrap();
        let mut streams: Vec<_> = [1, 3, width]
            .into_iter()
            .map(|chunk| {
                (
                    chunk,
                    TrellisStreamingDecoder::from_sparse_dem(&dem, config.clone()).unwrap(),
                )
            })
            .collect();
        for _ in 0..32 {
            let mut syndrome = vec![0; width];
            for detector in 0..12 {
                syndrome[detector * (width - 1) / 11] = if rng.random() { 0 } else { 7 };
            }
            let expected = batch.decode(&syndrome);
            for (chunk, stream) in &mut streams {
                stream.reset();
                let _ = stream.feed_prefix(&syndrome[..width / 2]);
                let _ = stream.advance();
                stream.reset();
                let mut commitments = Vec::new();
                let mut early = false;
                let actual = (|| {
                    for detectors in syndrome.chunks(*chunk) {
                        stream.feed_prefix(detectors)?;
                        let progress = stream.advance()?;
                        for &(logical, value) in &progress.newly_committed {
                            assert!(progress.committed_mask.get(logical));
                            assert_eq!(progress.committed.get(logical), value);
                            assert!(!commitments.iter().any(|&(prior, _)| prior == logical));
                            // Require a changing logical, not a never-toggled padding bit.
                            early |= logical < 4 && progress.columns_processed < 12;
                            commitments.push((logical, value));
                        }
                        assert!(stream.advance()?.newly_committed.is_empty());
                    }
                    stream.flush()
                })();
                match (&actual, &expected) {
                    (Ok(actual), Ok(expected)) => {
                        assert_bit_identical(actual, expected);
                        for (logical, value) in commitments {
                            assert_eq!(value, actual.predicted.get(logical));
                        }
                        saw_early_commitment |= early;
                        let mut fresh =
                            TrellisStreamingDecoder::from_sparse_dem(&dem, config.clone()).unwrap();
                        fresh.feed_dense(&syndrome).unwrap();
                        assert_bit_identical(actual, &fresh.flush().unwrap());
                        assert_bit_identical(actual, &stream.flush().unwrap());
                    }
                    (Err(actual), Err(expected)) => assert_no_path(actual, expected),
                    _ => panic!("batch/stream mismatch: {actual:?}, {expected:?}"),
                }
            }
        }
    }
    assert!(
        saw_early_commitment,
        "must commit a toggled logical before the last column"
    );
}

#[test]
fn forced_future_detector_delays_first_column() {
    let text = "error(1) D1\nerror(0.1) D0 L0\nerror(0.2) D1";
    let config = TrellisConfig {
        column_order: Some(vec![0, 1, 2]),
        ..TrellisConfig::default()
    };
    let mut batch = TrellisDecoder::from_dem_str(text, config.clone()).unwrap();
    let mut stream = TrellisStreamingDecoder::from_dem_str(text, config).unwrap();
    assert_eq!(stream.column_lookahead(), &[1, 0]);
    for syndrome in [[0, 0], [0, 1], [1, 0], [1, 1]] {
        stream.reset();
        assert_eq!(stream.advance().unwrap().columns_processed, 0);
        stream.feed_prefix(&syndrome[..1]).unwrap();
        assert_eq!(stream.advance().unwrap().columns_processed, 0);
        stream.feed_prefix(&syndrome[1..]).unwrap();
        assert_eq!(stream.advance().unwrap().columns_processed, 2);
        assert_bit_identical(&stream.flush().unwrap(), &batch.decode(&syndrome).unwrap());
    }
}

#[test]
fn no_path_errors_match_and_persist_until_reset() {
    for (text, syndrome, fails_on_feed) in [
        ("error(0.1) D0 D1 L0", [1, 0], false),
        ("error(0.1) D0 L0\ndetector D1", [0, 1], true),
        ("error(0.1) D0 L0\nerror(1) D1", [0, 0], true),
    ] {
        let config = TrellisConfig::default();
        let expected = TrellisDecoder::from_dem_str(text, config.clone())
            .unwrap()
            .decode(&syndrome)
            .unwrap_err();
        let mut stream = TrellisStreamingDecoder::from_dem_str(text, config.clone()).unwrap();
        stream.feed_prefix(&syndrome[..1]).unwrap();
        stream.advance().unwrap();
        let feed = stream.feed_prefix(&syndrome[1..]);
        if fails_on_feed {
            assert_no_path(&feed.unwrap_err(), &expected);
        } else {
            feed.unwrap();
        }
        assert_no_path(&stream.advance().unwrap_err(), &expected);
        assert_no_path(&stream.advance().unwrap_err(), &expected);
        assert_no_path(&stream.flush().unwrap_err(), &expected);
        stream.reset();
        let valid = if text.contains("error(1)") {
            [0, 1]
        } else {
            [0, 0]
        };
        stream.feed_dense(&valid).unwrap();
        let mut fresh = TrellisStreamingDecoder::from_dem_str(text, config).unwrap();
        fresh.feed_dense(&valid).unwrap();
        assert_bit_identical(&stream.flush().unwrap(), &fresh.flush().unwrap());
    }
}

#[test]
fn late_pruning_can_create_unanimity_after_last_toggle() {
    let text = "error(0.4) L0\nerror(0.5) D0 D1\nerror(0.1) D0 D1\nerror(0.1) D2";
    let config = TrellisConfig {
        k: 2,
        score_alpha: 0.0,
        ..TrellisConfig::default()
    };
    let mut stream = TrellisStreamingDecoder::from_dem_str(text, config.clone()).unwrap();
    let first = stream.advance().unwrap();
    assert_eq!(first.columns_processed, 1);
    assert!(!first.committed_mask.get(0));
    stream.feed_prefix(&[0, 0]).unwrap();
    let middle = stream.advance().unwrap();
    assert_eq!(middle.columns_processed, 3);
    assert_eq!(middle.newly_committed, vec![(0, false)]);
    stream.feed_prefix(&[0]).unwrap();
    let result = stream.flush().unwrap();
    let expected = TrellisDecoder::from_dem_str(text, config)
        .unwrap()
        .decode(&[0, 0, 0])
        .unwrap();
    assert_bit_identical(&result, &expected);
}

#[test]
fn empty_models_forced_bits_and_dimensions() {
    for text in ["", "error(1) L130", "error(0.3) L0"] {
        let config = TrellisConfig::default();
        let mut stream = TrellisStreamingDecoder::from_dem_str(text, config.clone()).unwrap();
        let progress = stream.advance().unwrap();
        if text == "error(1) L130" {
            assert_eq!(progress.newly_committed.len(), 131);
            assert!(progress.committed.get(130));
            assert_eq!(progress.committed_mask.count_ones(), 131);
        }
        let expected = TrellisDecoder::from_dem_str(text, config)
            .unwrap()
            .decode(&[])
            .unwrap();
        assert_bit_identical(&stream.flush().unwrap(), &expected);
    }
    let mut stream =
        TrellisStreamingDecoder::from_dem_str("error(0.1) D0", TrellisConfig::default()).unwrap();
    assert!(matches!(
        stream.flush(),
        Err(DecoderError::InvalidDimensions {
            expected: 1,
            actual: 0
        })
    ));
    assert!(matches!(
        stream.feed_prefix(&[0, 0]),
        Err(DecoderError::InvalidDimensions {
            expected: 1,
            actual: 2
        })
    ));
    stream.feed_prefix(&[]).unwrap();
    stream.feed_prefix(&[9]).unwrap();
    assert!(matches!(
        stream.feed_prefix(&[0]),
        Err(DecoderError::InvalidDimensions {
            expected: 1,
            actual: 2
        })
    ));
    stream.flush().unwrap();
}

#[test]
fn rejects_bp_nary_and_integer_metrics() {
    let dem = SparseDem::from_dem_str("error(0.1) D0 L0").unwrap();
    for config in [
        TrellisConfig {
            bp_score_iterations: 1,
            ..TrellisConfig::default()
        },
        TrellisConfig {
            k: usize::MAX,
            delta: f64::INFINITY,
            bp_score_iterations: 1,
            ..TrellisConfig::default()
        },
        TrellisConfig {
            metric_mode: MetricMode::MaxLogInt,
            ..TrellisConfig::default()
        },
    ] {
        let error = TrellisStreamingDecoder::from_sparse_dem(&dem, config.clone()).unwrap_err();
        assert!(matches!(error, DecoderError::InvalidConfiguration(_)));
        assert!(
            error
                .to_string()
                .contains(if config.bp_score_iterations > 0 {
                    "whole syndrome"
                } else {
                    "streaming v1 supports the LogSumExpFloat metric"
                })
        );
    }
    let model = FactorModel::new(
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
    let error =
        TrellisStreamingDecoder::from_factor_model(&model, TrellisConfig::default()).unwrap_err();
    assert!(matches!(error, DecoderError::InvalidConfiguration(_)));
    assert!(
        error
            .to_string()
            .contains("streaming v1 supports the binary float kernel")
    );
}

#[test]
fn word_boundaries_and_nonmonotone_column_detectors() {
    let dem = SparseDem::from_dem_str(
        "error(1) D64 L129\nerror(0.2) D65 L65\nerror(0.3) D0 L0\nerror(0.4) D64 L64",
    )
    .unwrap();
    let config = TrellisConfig::default();
    let model = FactorModel::try_from(&dem).unwrap();
    let mut stream = TrellisStreamingDecoder::from_factor_model(&model, config.clone()).unwrap();
    let mut syndrome = vec![0; 66];
    syndrome[0] = 1;
    syndrome[64] = 1;
    syndrome[65] = 2;
    stream.feed_prefix(&syndrome[..64]).unwrap();
    let initial = stream.advance().unwrap();
    assert_eq!(initial.columns_processed, 0);
    assert!(initial.committed_mask.get(129));
    assert!(initial.committed.get(129));
    stream.feed_prefix(&syndrome[64..65]).unwrap();
    assert_eq!(stream.advance().unwrap().columns_processed, 0);
    stream.feed_prefix(&syndrome[65..]).unwrap();
    assert_eq!(stream.advance().unwrap().columns_processed, 3);
    let expected = TrellisDecoder::from_sparse_dem(&dem, config)
        .unwrap()
        .decode(&syndrome)
        .unwrap();
    assert_bit_identical(&stream.flush().unwrap(), &expected);
}

#[test]
fn stored_failure_precedes_incomplete_flush() {
    assert_stored_failure(&[0]);
}

#[test]
fn stored_failure_precedes_overflowing_feed() {
    assert_stored_failure(&[0, 0]);
}

fn assert_stored_failure(later: &[u8]) {
    let text = "detector D0\nerror(0.1) D1 L0";
    let config = TrellisConfig::default();
    let expected = TrellisDecoder::from_dem_str(text, config.clone())
        .unwrap()
        .decode(&[1, 0])
        .unwrap_err();
    let mut stream = TrellisStreamingDecoder::from_dem_str(text, config).unwrap();
    assert_no_path(&stream.feed_prefix(&[1]).unwrap_err(), &expected);
    assert_no_path(&stream.feed_prefix(later).unwrap_err(), &expected);
    assert_no_path(&stream.flush().unwrap_err(), &expected);
}

#[test]
fn dense_requires_a_complete_fresh_shot_and_flush_exposes_commitments() {
    let mut stream = TrellisStreamingDecoder::from_dem_str(
        "error(0.1) D0 L0\nerror(0.1) D1 L1",
        TrellisConfig::default(),
    )
    .unwrap();
    assert!(matches!(
        stream.feed_dense(&[1]),
        Err(DecoderError::InvalidDimensions { .. })
    ));
    stream.feed_prefix(&[1]).unwrap();
    assert!(matches!(
        stream.feed_dense(&[0]),
        Err(DecoderError::InvalidDimensions { .. })
    ));
    assert!(matches!(
        stream.feed_dense(&[1, 0]),
        Err(DecoderError::InvalidDimensions { .. })
    ));
    stream.reset();
    assert_eq!(stream.committed().1.count_ones(), 0);
    stream.feed_dense(&[1, 0]).unwrap();
    assert_eq!(stream.committed().1.count_ones(), 0);
    let result = stream.flush().unwrap();
    let (values, mask) = stream.committed();
    assert_eq!(mask.count_ones(), 2);
    assert_eq!(values, result.predicted);
}
