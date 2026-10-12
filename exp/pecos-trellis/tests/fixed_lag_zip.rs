// Copyright 2026 The PECOS Developers
// SPDX-License-Identifier: Apache-2.0
use pecos_trellis::{StreamingMode, TrellisConfig, TrellisStreamingDecoder};
use std::fmt::Write;

#[test]
fn prediction_only_requires_reset_before_evidence() {
    let mut decoder = TrellisStreamingDecoder::from_dem_str(
        "error(0.1) D0 L0\nerror(0.2) D1\nerror(0.3) D2",
        TrellisConfig::default(),
    )
    .unwrap();
    decoder.feed_dense(&[1, 0, 0]).unwrap();
    let prediction = decoder.flush_prediction_without_evidence().unwrap();
    assert!(decoder.flush().is_err());
    assert!(decoder.advance_prediction().is_err());
    decoder.reset();
    decoder.feed_dense(&[1, 0, 0]).unwrap();
    assert_eq!(decoder.flush().unwrap().predicted, prediction);
}

#[test]
fn fixed_lag_matches_original_zip_native_golden_predictions_and_schedule() {
    let doc: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/fixed_lag_zip_v22.json")).unwrap();
    let mut dem = String::new();
    for mechanism in doc["model"]["mechanisms"].as_array().unwrap() {
        write!(dem, "error({})", mechanism[0].as_f64().unwrap()).unwrap();
        for d in mechanism[1].as_array().unwrap() {
            write!(dem, " D{}", d.as_u64().unwrap()).unwrap();
        }
        for o in mechanism[2].as_array().unwrap() {
            write!(dem, " L{}", o.as_u64().unwrap()).unwrap();
        }
        dem.push('\n');
    }
    for case in doc["cases"].as_array().unwrap() {
        for omit_evidence in [false, true] {
            let mut decoder = TrellisStreamingDecoder::from_dem_str(
                &dem,
                TrellisConfig {
                    k: usize::try_from(case["k"].as_u64().unwrap()).unwrap(),
                    delta: case["delta"].as_f64().unwrap(),
                    score_alpha: 0.8,
                    merge_indistinguishable: false,
                    ..TrellisConfig::default()
                },
            )
            .unwrap();
            decoder
                .set_streaming_mode(StreamingMode::FixedLag {
                    detector_round_bounds: vec![0, 2, 4, 6],
                    window_size: 2,
                })
                .unwrap();
            let syndrome = case["syndrome"].as_u64().unwrap();
            let bits: Vec<_> = (0..6).map(|d| ((syndrome >> d) & 1) as u8).collect();
            let mut columns = Vec::new();
            let mut failed = false;
            for chunk in bits.chunks(2) {
                decoder.feed_prefix(chunk).unwrap();
                let advanced = if omit_evidence {
                    decoder.advance_prediction_without_evidence()
                } else {
                    decoder.advance_prediction()
                };
                if let Ok(count) = advanced {
                    columns.push(count);
                } else {
                    failed = true;
                    break;
                }
            }
            let prediction = if failed {
                None
            } else {
                Some(
                    if omit_evidence {
                        decoder.flush_prediction_without_evidence()
                    } else {
                        decoder.flush_prediction()
                    }
                    .unwrap()
                    .words()[0],
                )
            };
            assert_eq!(prediction, case["prediction"].as_u64(), "{case}");
            assert_eq!(serde_json::json!(columns), case["columns"], "{case}");
        }
    }
}
