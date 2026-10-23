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

use pecos_frontier::recovery::{
    FallbackPolicy, FrontierRecoveryDecoder, FrontierRecoveryOutcome, NoPathAction, NoPathCause,
    RecoveryOptions,
};
use pecos_frontier::{FrontierConfig, FrontierDecoder, MetricMode, SparseDem};

const DEM: &str = "error(0.4) D0\nerror(0.4) D1\nerror(0.1) D0 D1 D2 L70";
fn config(k: usize, delta: f64) -> FrontierConfig {
    FrontierConfig {
        k,
        delta,
        score_alpha: 0.0,
        column_order: Some(vec![0, 1, 2]),
        ..Default::default()
    }
}
fn options(action: NoPathAction, max: Option<f64>) -> RecoveryOptions {
    RecoveryOptions {
        delta_max: max,
        on_no_path: action,
    }
}

#[test]
fn bounded_ladder_keeps_width_and_restarts_without_mutating_base() {
    let dem = SparseDem::from_dem_str(DEM).unwrap();
    let mut decoder = FrontierRecoveryDecoder::from_sparse_dem(&dem, config(1, 0.0), None).unwrap();
    let FrontierRecoveryOutcome::NoPath(report) = decoder
        .decode(&[0, 0, 1], options(NoPathAction::Report, Some(3.0)))
        .unwrap()
    else {
        panic!("width-limited shot must fail");
    };
    assert_eq!(report.deltas_tried, [0.0, 1.0, 2.0, 3.0]);
    assert_eq!(report.cause, NoPathCause::Exhausted);
    assert!(report.dropped_states > 0);
    let mut transitions = 0;
    for delta in &report.deltas_tried {
        let mut reference = FrontierDecoder::from_sparse_dem(&dem, config(1, *delta)).unwrap();
        let pecos_frontier::FrontierDecodeAttempt::NoPath { transitions: n, .. } =
            reference.decode_attempt(&[0, 0, 1])
        else {
            panic!("same width");
        };
        transitions += n;
    }
    assert_eq!(report.transitions, transitions);
    let FrontierRecoveryOutcome::NoPath(base) = decoder
        .decode(&[0, 0, 1], options(NoPathAction::Report, None))
        .unwrap()
    else {
        panic!("base unchanged");
    };
    assert_eq!(base.deltas_tried, [0.0]);
    assert!(
        decoder
            .decode(&[0, 0, 1], RecoveryOptions::default())
            .is_err()
    );
}

#[test]
fn successful_retry_stops_and_matches_independent_rung() {
    let dem = SparseDem::from_dem_str(DEM).unwrap();
    for mode in [MetricMode::LogSumExpFloat, MetricMode::MaxLogInt] {
        let mut cfg = config(8, 0.0);
        cfg.metric_mode = mode;
        let mut decoder =
            FrontierRecoveryDecoder::from_sparse_dem(&dem, cfg.clone(), None).unwrap();
        let FrontierRecoveryOutcome::Decoded {
            result,
            deltas_tried,
        } = decoder
            .decode(&[0, 0, 1], options(NoPathAction::Raise, Some(32.0)))
            .unwrap()
        else {
            panic!("retry should decode");
        };
        assert!(deltas_tried.len() > 1);
        assert!(*deltas_tried.last().unwrap() < 32.0);
        assert_eq!(
            result.escalation_rungs_used as usize,
            deltas_tried.len() - 1
        );
        cfg.delta = *deltas_tried.last().unwrap();
        let direct = FrontierDecoder::from_sparse_dem(&dem, cfg)
            .unwrap()
            .decode(&[0, 0, 1])
            .unwrap();
        assert_eq!(result.predicted, direct.predicted);
        assert_eq!(result.log_evidence, direct.log_evidence);
        assert!(result.predicted.get(70));
    }
}

#[test]
fn bp_prepares_once_for_all_retries() {
    let dem = SparseDem::from_dem_str(DEM).unwrap();
    let mut cfg = config(8, 0.0);
    cfg.bp_score_iterations = 2;
    let mut decoder = FrontierRecoveryDecoder::from_sparse_dem(&dem, cfg, None).unwrap();
    let FrontierRecoveryOutcome::Decoded {
        result,
        deltas_tried,
    } = decoder
        .decode(&[0, 0, 1], options(NoPathAction::Raise, Some(32.0)))
        .unwrap()
    else {
        panic!("decode");
    };
    assert!(deltas_tried.len() > 1);
    assert_eq!(result.bp_runs, 1);
}

#[test]
fn proven_impossible_skips_retry_and_fallback_is_honest() {
    for (dem, shot, expected_attempts) in [
        ("error(0.1) D0 D1", vec![1, 0], 1),
        ("error(1) D0 L70", vec![0], 0),
    ] {
        let dem = SparseDem::from_dem_str(dem).unwrap();
        for fallback in [FallbackPolicy::FeasibleThenZero, FallbackPolicy::Zero] {
            let mut decoder = FrontierRecoveryDecoder::from_sparse_dem(
                &dem,
                FrontierConfig::default(),
                Some(fallback),
            )
            .unwrap();
            let FrontierRecoveryOutcome::Fallback {
                predicted,
                syndrome_consistent,
                mechanisms,
                no_path,
                ..
            } = decoder
                .decode(&shot, options(NoPathAction::Fallback, Some(100.0)))
                .unwrap()
            else {
                panic!("fallback");
            };
            assert!(predicted.is_zero());
            assert_eq!(no_path.deltas_tried.len(), expected_attempts);
            assert!(mechanisms.is_none());
            assert_eq!(
                syndrome_consistent,
                (fallback == FallbackPolicy::FeasibleThenZero).then_some(false)
            );
        }
    }
}

#[test]
fn feasible_fallback_is_verified_and_shared_across_workers() {
    let dem = SparseDem::from_dem_str(DEM).unwrap();
    let decoder = FrontierRecoveryDecoder::from_sparse_dem(
        &dem,
        config(1, 0.0),
        Some(FallbackPolicy::FeasibleThenZero),
    )
    .unwrap();
    let shots = vec![vec![0, 0, 0], vec![0, 0, 1], vec![1, 1, 1], vec![0, 0, 1]];
    for workers in [1, 4] {
        let results = decoder
            .decode_batch(&shots, workers, options(NoPathAction::Fallback, Some(3.0)))
            .unwrap();
        for (shot, result) in shots.iter().zip(results) {
            match result.unwrap() {
                FrontierRecoveryOutcome::Decoded { .. } => {}
                FrontierRecoveryOutcome::Fallback {
                    mechanisms,
                    syndrome_consistent,
                    predicted,
                    ..
                } => {
                    assert_eq!(syndrome_consistent, Some(true));
                    let mut check = vec![0u8; 3];
                    let mut logical = pecos_frontier::ObsMask::new();
                    for i in mechanisms.unwrap() {
                        for &d in &dem.mechanisms[i].1 {
                            check[d as usize] ^= 1;
                        }
                        for &l in &dem.mechanisms[i].2 {
                            logical.toggle(l as usize);
                        }
                    }
                    assert_eq!(&check, shot);
                    assert_eq!(predicted, logical);
                }
                FrontierRecoveryOutcome::NoPath(_) => panic!("always-answer mode"),
            }
        }
    }
}

#[test]
fn policies_validate_even_for_empty_batches_and_never_hide_bad_input() {
    let dem = SparseDem::from_dem_str(DEM).unwrap();
    let mut decoder =
        FrontierRecoveryDecoder::from_sparse_dem(&dem, config(1, 2.0), Some(FallbackPolicy::Zero))
            .unwrap();
    for max in [f64::NAN, f64::INFINITY, -1.0, 1.0] {
        assert!(
            decoder
                .decode_batch(&[], 1, options(NoPathAction::Report, Some(max)))
                .is_err()
        );
    }
    assert!(
        decoder
            .decode(&[0], options(NoPathAction::Fallback, None))
            .is_err()
    );
    let results = decoder
        .decode_batch(
            &[vec![0], vec![0, 0, 1]],
            2,
            options(NoPathAction::Fallback, None),
        )
        .unwrap();
    assert!(results[0].is_err());
    assert!(results[1].is_ok());
    let decoder = FrontierRecoveryDecoder::from_sparse_dem(&dem, config(1, 2.0), None).unwrap();
    assert!(
        decoder
            .decode_batch(&[], 1, options(NoPathAction::Fallback, None))
            .is_err()
    );
}

#[test]
fn configured_observable_adapter_preserves_fallback_and_reporting() {
    use pecos_decoder_core::ObservableDecoder;
    let dem = SparseDem::from_dem_str(DEM).unwrap();
    for action in [NoPathAction::Report, NoPathAction::Fallback] {
        let mut adapter = FrontierRecoveryDecoder::from_sparse_dem(
            &dem,
            config(1, 0.0),
            Some(FallbackPolicy::FeasibleThenZero),
        )
        .unwrap()
        .with_options(options(action, Some(3.0)))
        .unwrap();
        assert_eq!(adapter.num_detectors(), Some(3));
        let outcome = adapter.decode_outcome(&[0, 0, 1]).unwrap();
        assert_eq!(outcome.deltas_tried, [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(outcome.fallback_used, action == NoPathAction::Fallback);
        if action == NoPathAction::Fallback {
            assert_eq!(outcome.syndrome_consistent, Some(true));
            assert_eq!(
                outcome
                    .prediction
                    .unwrap()
                    .iter_set_bits()
                    .collect::<Vec<_>>(),
                [70]
            );
            assert!(adapter.decode_obs(&[0, 0, 1]).is_ok());
        } else {
            assert!(outcome.prediction.is_none());
            assert!(adapter.decode_obs(&[0, 0, 1]).is_err());
        }
        assert!(adapter.decode_outcome(&[]).is_err());
    }
}
