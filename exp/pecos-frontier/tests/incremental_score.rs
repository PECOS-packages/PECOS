// Copyright 2026 The PECOS Developers
// SPDX-License-Identifier: Apache-2.0

use pecos_frontier::{
    FrontierCommittee, FrontierConfig, FrontierDecoder, FrontierResult, MetricMode, SparseDem,
};
use rand::{RngExt, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

fn agree(a: &FrontierResult, b: &FrontierResult, context: &str) {
    assert_eq!(a.status, b.status, "{context}");
    assert_eq!(a.predicted, b.predicted, "{context}");
    assert!(
        (a.log_evidence - b.log_evidence).abs() <= 1e-6,
        "{context}: evidence {} vs {}",
        a.log_evidence,
        b.log_evidence
    );
}

#[test]
fn public_d5_d7_pruned_corpus_and_committee_agree() {
    for (distance, source, shots) in [
        (
            5,
            include_str!("../../../examples/surface_code_circuits/surface_code_d5_z_stim.dem"),
            200,
        ),
        (
            7,
            include_str!("../../../examples/surface_code_circuits/surface_code_d7_z_stim.dem"),
            50,
        ),
    ] {
        let dem = SparseDem::from_dem_str(source).unwrap();
        for (k, delta) in [(16, 100.0), (512, 12.0)] {
            let cfg = FrontierConfig {
                k,
                delta,
                ..FrontierConfig::default()
            };
            let mut summed = FrontierDecoder::from_sparse_dem(&dem, cfg.clone()).unwrap();
            let mut summed_committee =
                FrontierCommittee::from_sparse_dem(&dem, cfg.clone()).unwrap();
            let cfg = FrontierConfig {
                incremental_score: true,
                ..cfg
            };
            let mut carried = FrontierDecoder::from_sparse_dem(&dem, cfg.clone()).unwrap();
            let mut carried_committee = FrontierCommittee::from_sparse_dem(&dem, cfg).unwrap();
            let mut dropped = 0;
            let mut committee_dropped = 0;
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(0x5eed);
            for index in 0..shots {
                let mut shot = vec![0; dem.num_detectors];
                for &(probability, ref detectors, _) in &dem.mechanisms {
                    if rng.random::<f64>() < probability {
                        for &detector in detectors {
                            shot[detector as usize] ^= 1;
                        }
                    }
                }
                let context =
                    format!("d{distance} k={k} delta={delta} shot={index} syndrome={shot:?}");
                match (summed.decode(&shot), carried.decode(&shot)) {
                    (Ok(a), Ok(b)) => {
                        agree(&a, &b, &context);
                        dropped += a.dropped_states;
                    }
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string(), "{context}"),
                    (a, b) => panic!("{context}: status mismatch {a:?} vs {b:?}"),
                }
                match (
                    summed_committee.decode(&shot),
                    carried_committee.decode(&shot),
                ) {
                    (Ok(a), Ok(b)) => {
                        agree(&a.selected, &b.selected, &context);
                        assert_eq!(a.forward.status, b.forward.status, "{context}");
                        assert_eq!(a.backward.status, b.backward.status, "{context}");
                        for (left, right) in [
                            (a.forward.log_evidence, b.forward.log_evidence),
                            (a.backward.log_evidence, b.backward.log_evidence),
                        ] {
                            assert!(
                                left.to_bits() == right.to_bits() || (left - right).abs() <= 1e-6,
                                "{context}: committee evidence {left} vs {right}"
                            );
                        }
                        committee_dropped += a.selected.dropped_states;
                    }
                    (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string(), "{context}"),
                    (a, b) => panic!("{context}: committee status mismatch {a:?} vs {b:?}"),
                }
            }
            assert!(dropped > 0, "d{distance} k={k}");
            assert!(committee_dropped > 0, "committee d{distance} k={k}");
        }
    }
}

#[test]
fn committee_preserves_integer_rejection_precedence() {
    let dem = SparseDem::from_dem_str("error(0.1) D0 L0").unwrap();
    let error = FrontierCommittee::from_sparse_dem(
        &dem,
        FrontierConfig {
            incremental_score: true,
            metric_mode: MetricMode::MaxLogInt,
            ..FrontierConfig::default()
        },
    )
    .unwrap_err();
    assert!(matches!(
        error,
        pecos_frontier::DecoderError::InvalidConfiguration(_)
    ));
    assert!(
        error
            .to_string()
            .contains("FrontierCommittee arbitration ranks on coset-mass posteriors")
    );
}

#[test]
fn committee_validates_incremental_in_both_legs() {
    for source in [
        "error(0.1) D0\nerror(1e-17) D0",
        "error(1e-17) D0\nerror(0.1) D0",
    ] {
        let dem = SparseDem::from_dem_str(source).unwrap();
        FrontierCommittee::from_sparse_dem(&dem, FrontierConfig::default()).unwrap();
        let error = FrontierCommittee::from_sparse_dem(
            &dem,
            FrontierConfig {
                incremental_score: true,
                ..FrontierConfig::default()
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("incremental_score"));
        assert!(error.to_string().contains("leave incremental_score off"));
    }
}

#[test]
fn recovery_delta_ladder_carries_and_restarts() {
    use pecos_frontier::recovery::{
        FrontierRecoveryDecoder, FrontierRecoveryOutcome, RecoveryOptions,
    };
    let dem = SparseDem::from_dem_str("error(0.4) D0\nerror(0.1) D0 D1 L0").unwrap();
    let mut results = Vec::new();
    for incremental_score in [false, true] {
        let config = FrontierConfig {
            k: 16,
            delta: 0.01,
            score_alpha: 0.1,
            incremental_score,
            ..FrontierConfig::default()
        };
        let mut decoder = FrontierRecoveryDecoder::from_sparse_dem(&dem, config, None).unwrap();
        let FrontierRecoveryOutcome::Decoded {
            result,
            deltas_tried,
        } = decoder
            .decode(
                &[0, 1],
                RecoveryOptions {
                    delta_max: Some(100.0),
                    ..RecoveryOptions::default()
                },
            )
            .unwrap()
        else {
            panic!("recovered");
        };
        assert!(deltas_tried.len() > 1);
        results.push((result, deltas_tried));
    }
    agree(&results[0].0, &results[1].0, "delta ladder");
    assert_eq!(results[0].1, results[1].1);
}

#[test]
fn incremental_windowed_and_bounded_committee_agree() {
    use pecos_frontier::windowed::{WindowConfig, WindowedDecoder, ZeroSyndromePolicy};
    let dem = SparseDem::from_dem_str(
        "error(0.2) D0 L0\nerror(0.1) D0 L1\nerror(0.3) D0 L0 L1\nerror(0.15) D0",
    )
    .unwrap();
    let cfg = FrontierConfig {
        k: 2,
        delta: 100.0,
        incremental_score: true,
        ..FrontierConfig::default()
    };
    let window = WindowConfig {
        detector_blocks: vec![vec![0]],
        window_blocks: 2,
        stride_blocks: 1,
        zero_policy: ZeroSyndromePolicy::Decode,
        detector_free_error_owners: std::collections::BTreeMap::new(),
    };
    let mut windowed = WindowedDecoder::from_sparse_dem(&dem, window, cfg.clone(), 32).unwrap();
    let mut direct = FrontierCommittee::from_sparse_dem(&dem, cfg.clone()).unwrap();
    let mut bounded = FrontierCommittee::from_sparse_dem(&dem, cfg)
        .unwrap()
        .into_resumable();
    for shot in [[0], [1]] {
        let expected = direct.decode(&shot).unwrap();
        let actual = windowed.decode(&shot).unwrap();
        assert_eq!(actual.predicted, expected.selected.predicted);
        assert_eq!(actual.windows[0].committee.as_ref().unwrap(), &expected);
        bounded.begin(&shot).unwrap();
        while bounded.progress() != Some(pecos_frontier::FrontierBoundedProgress::Ready) {
            bounded.step(1).unwrap();
        }
        assert_eq!(bounded.finish().unwrap(), expected);
    }
}

#[test]
fn recovery_constructor_honours_incremental_score() {
    use pecos_frontier::recovery::FrontierRecoveryDecoder;
    let dem = SparseDem::from_dem_str("error(0.1) D0 L0\nerror(1e-17) D0").unwrap();
    FrontierRecoveryDecoder::from_sparse_dem(&dem, FrontierConfig::default(), None).unwrap();
    let error = FrontierRecoveryDecoder::from_sparse_dem(
        &dem,
        FrontierConfig {
            incremental_score: true,
            ..FrontierConfig::default()
        },
        None,
    )
    .unwrap_err();
    assert!(matches!(
        error,
        pecos_frontier::DecoderError::InvalidConfiguration(_)
    ));
    assert!(error.to_string().contains("incremental_score"), "{error}");
}

#[test]
fn windowed_constructor_honours_incremental_score() {
    use pecos_frontier::windowed::{WindowConfig, WindowedDecoder, ZeroSyndromePolicy};
    let dem = SparseDem::from_dem_str("error(0.1) D0 L0\nerror(1e-17) D0").unwrap();
    let window = WindowConfig {
        detector_blocks: vec![vec![0]],
        window_blocks: 2,
        stride_blocks: 1,
        zero_policy: ZeroSyndromePolicy::Decode,
        detector_free_error_owners: std::collections::BTreeMap::new(),
    };
    WindowedDecoder::from_sparse_dem(&dem, window.clone(), FrontierConfig::default(), 32).unwrap();
    let error = WindowedDecoder::from_sparse_dem(
        &dem,
        window,
        FrontierConfig {
            incremental_score: true,
            ..FrontierConfig::default()
        },
        32,
    )
    .err()
    .expect("incremental model must be rejected");
    assert!(matches!(
        error,
        pecos_frontier::DecoderError::InvalidConfiguration(_)
    ));
    assert!(error.to_string().contains("incremental_score"), "{error}");
}
