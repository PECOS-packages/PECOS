// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

use pecos_trellis::factor::{Factor, FactorModel, Outcome};
use pecos_trellis::{
    DecoderError, MetricMode, ObsMask, PruneParams, SparseDem, TrellisConfig, TrellisDecodeAttempt,
    TrellisDecoder, TrellisPrepared,
};
use serde::Deserialize;
use std::collections::BTreeMap;

type Mechanism = (f64, Vec<u32>, Vec<u32>);

#[derive(Deserialize)]
struct Fixture {
    name: String,
    num_detectors: usize,
    num_observables: usize,
    #[serde(default)]
    mechanisms: Vec<Mechanism>,
    #[serde(default)]
    factors: Vec<Vec<Mechanism>>,
}

fn fixture(name: &str, config: TrellisConfig) -> TrellisDecoder {
    let fixtures: BTreeMap<String, Vec<Fixture>> =
        serde_json::from_str(include_str!("fixtures/models.json")).unwrap();
    let model = fixtures
        .values()
        .flatten()
        .find(|f| f.name == name)
        .unwrap();
    if model.factors.is_empty() {
        TrellisDecoder::from_sparse_dem(
            &SparseDem {
                mechanisms: model.mechanisms.clone(),
                num_detectors: model.num_detectors,
                num_observables: model.num_observables,
                detector_coords: BTreeMap::new(),
            },
            config,
        )
        .unwrap()
    } else {
        let factors = model
            .factors
            .iter()
            .map(|outcomes| Factor {
                outcomes: outcomes
                    .iter()
                    .map(|(p, d, l)| Outcome {
                        probability: *p,
                        detectors: d.clone(),
                        observables: l.clone(),
                    })
                    .collect(),
            })
            .collect();
        TrellisDecoder::from_factor_model(
            &FactorModel::new(factors, model.num_detectors, model.num_observables).unwrap(),
            config,
        )
        .unwrap()
    }
}

fn config(integer: bool) -> TrellisConfig {
    TrellisConfig {
        k: 16,
        delta: 100.0,
        score_alpha: 0.0,
        bp_score_iterations: 0,
        merge_indistinguishable: false,
        metric_mode: if integer {
            MetricMode::MaxLogInt
        } else {
            MetricMode::LogSumExpFloat
        },
        ..TrellisConfig::default()
    }
}

// Debug's round-trippable float representations retain the exact values,
// including signed zero; exclude only the nondeterministic clock.
fn fingerprint(attempt: TrellisDecodeAttempt) -> String {
    match attempt {
        TrellisDecodeAttempt::Success(mut result) => {
            result.bp_seconds = 0.0;
            format!("{result:?}")
        }
        TrellisDecodeAttempt::NoPath {
            error,
            transitions,
            dropped_states,
            ..
        } => format!("{error:?}/{transitions}/{dropped_states}"),
        TrellisDecodeAttempt::Error(error) => panic!("unexpected error: {error}"),
    }
}

fn override_guard(nary: bool, integer: bool, change_k: bool) {
    let (name, shot) = if nary {
        ("three_route_collision", vec![])
    } else {
        ("overpruning_ladder", vec![0; 4])
    };
    let base = config(integer);
    let mut overridden = base.clone();
    if change_k {
        overridden.k = 1;
    } else {
        overridden.delta = 0.01;
    }
    if integer && !change_k {
        assert_ne!(
            (base.delta * 1024.0).round().to_bits(),
            (overridden.delta * 1024.0).round().to_bits()
        );
    }
    let mut decoder = fixture(name, base);
    let original = fingerprint(decoder.decode_attempt(&shot));
    assert!(matches!(
        decoder.prepare(&shot).unwrap(),
        TrellisPrepared::Ready { .. }
    ));
    let actual = fingerprint(decoder.attempt(PruneParams {
        k: overridden.k,
        delta: overridden.delta,
    }));
    let expected = fingerprint(fixture(name, overridden).decode_attempt(&shot));
    assert_ne!(
        actual, original,
        "override must change outcome or telemetry"
    );
    assert_eq!(
        actual, expected,
        "override must match independently constructed decoder"
    );
}

macro_rules! override_test {
    ($name:ident, $nary:expr, $integer:expr, $k:expr) => {
        #[test]
        fn $name() {
            override_guard($nary, $integer, $k);
        }
    };
}
override_test!(binary_float_k_override, false, false, true);
override_test!(binary_float_delta_override, false, false, false);
override_test!(nary_float_k_override, true, false, true);
override_test!(nary_float_delta_override, true, false, false);
override_test!(binary_maxlog_k_override, false, true, true);
override_test!(binary_maxlog_delta_override, false, true, false);
override_test!(nary_maxlog_k_override, true, true, true);
override_test!(nary_maxlog_delta_override, true, true, false);

#[test]
fn no_path_drops_in_all_four_arms() {
    for integer in [false, true] {
        for nary in [false, true] {
            let (name, positive, zero) = if nary {
                (
                    "nary_ladder_parity_locked",
                    vec![0, 0, 1, 1, 0],
                    vec![0, 0, 1, 0, 0],
                )
            } else {
                ("overpruning_ladder", vec![0, 0, 1, 0], vec![0, 0, 1, 0])
            };
            let mut narrow = config(integer);
            narrow.k = 1;
            match fixture(name, narrow).decode_attempt(&positive) {
                TrellisDecodeAttempt::NoPath { dropped_states, .. } => assert!(
                    dropped_states > 0,
                    "positive drops, nary={nary}, integer={integer}"
                ),
                _ => panic!("expected positive-drop no-path"),
            }
            let mut decoder = if nary {
                fixture(name, config(integer))
            } else {
                fixture("parity_locked_and_untouched", config(integer))
            };
            let zero = if nary { zero } else { vec![1, 0, 0] };
            match decoder.decode_attempt(&zero) {
                TrellisDecodeAttempt::NoPath { dropped_states, .. } => {
                    assert_eq!(dropped_states, 0);
                }
                _ => panic!("expected zero-drop no-path"),
            }
        }
    }
}

fn invalid(attempt: TrellisDecodeAttempt, rule: &str) {
    match attempt {
        TrellisDecodeAttempt::Error(DecoderError::InvalidConfiguration(message)) => {
            assert!(message.contains(rule), "{message:?} must name {rule:?}");
        }
        _ => panic!("expected InvalidConfiguration naming {rule}"),
    }
}

#[test]
fn parameter_rejection_matrix_precedes_readiness() {
    for integer in [false, true] {
        for name in ["overpruning_ladder", "three_route_collision"] {
            let mut decoder = fixture(name, config(integer));
            invalid(
                decoder.attempt(PruneParams { k: 0, delta: 100.0 }),
                "at least 1",
            );
            for delta in [f64::NAN, -1.0, f64::NEG_INFINITY] {
                invalid(
                    decoder.attempt(PruneParams { k: 8, delta }),
                    "non-negative and not NaN",
                );
            }
            if integer {
                invalid(
                    decoder.attempt(PruneParams {
                        k: 8,
                        delta: f64::INFINITY,
                    }),
                    "finite under maxlog_int",
                );
            } else {
                let shot = if name == "overpruning_ladder" {
                    vec![0; 4]
                } else {
                    vec![]
                };
                decoder.prepare(&shot).unwrap();
                assert!(matches!(
                    decoder.attempt(PruneParams {
                        k: 8,
                        delta: f64::INFINITY
                    }),
                    TrellisDecodeAttempt::Success(_)
                ));
            }
        }
    }
}

#[test]
fn bp_capabilities_and_refresh_count() {
    let exact = PruneParams {
        k: usize::MAX,
        delta: f64::INFINITY,
    };
    for name in ["overpruning_ladder", "three_route_collision"] {
        let mut cfg = config(false);
        cfg.k = exact.k;
        cfg.delta = exact.delta;
        cfg.bp_score_iterations = 5;
        let mut decoder = fixture(name, cfg);
        invalid(
            decoder.attempt(PruneParams { k: 8, delta: 100.0 }),
            "require a BP graph",
        );
        let shot = if name == "overpruning_ladder" {
            vec![0; 4]
        } else {
            vec![]
        };
        decoder.prepare(&shot).unwrap();
        for params in [
            PruneParams {
                k: 8,
                delta: f64::INFINITY,
            },
            PruneParams {
                k: usize::MAX,
                delta: 100.0,
            },
        ] {
            invalid(decoder.attempt(params), "require a BP graph");
        }

        assert!(matches!(
            decoder.attempt(exact),
            TrellisDecodeAttempt::Success(_)
        ));
        assert_eq!(decoder.bp_refreshes(), 0);
    }
    let mut cfg = config(false);
    cfg.bp_score_iterations = 5;
    let mut decoder = fixture("overpruning_ladder", cfg);
    assert_eq!(decoder.bp_refreshes(), 0);
    let TrellisPrepared::Ready { bp_ran, bp_seconds } = decoder.prepare(&[0; 4]).unwrap() else {
        panic!("ready")
    };
    assert!(bp_ran);
    assert_eq!(decoder.bp_refreshes(), 1);
    let TrellisDecodeAttempt::Success(result) = decoder.attempt(exact) else {
        panic!("decoded")
    };
    assert_eq!(result.bp_runs, 1);
    assert_eq!(result.bp_seconds.to_bits(), bp_seconds.to_bits());
    assert_eq!(decoder.bp_refreshes(), 1);
}

#[test]
fn repeated_attempts_and_next_shot_match_fresh_decoders() {
    for integer in [false, true] {
        for name in ["overpruning_ladder", "nary_ladder_parity_locked"] {
            let cfg = config(integer);
            let n = if name == "overpruning_ladder" { 4 } else { 5 };
            let a = vec![0; n];
            let mut b = a.clone();
            b[0] = 1;
            let mut decoder = fixture(name, cfg.clone());
            let p = decoder.prune_params();
            let q = PruneParams { k: 1, delta: 0.01 };
            decoder.prepare(&a).unwrap();
            let first = fingerprint(decoder.attempt(p));
            let _ = decoder.attempt(q);
            assert_eq!(first, fingerprint(decoder.attempt(p)));
            assert_eq!(
                first,
                fingerprint(fixture(name, cfg.clone()).decode_attempt(&a))
            );
            assert_eq!(first, fingerprint(decoder.clone().attempt(p)));
            decoder.prepare(&b).unwrap();
            assert_eq!(
                fingerprint(decoder.attempt(p)),
                fingerprint(fixture(name, cfg).decode_attempt(&b))
            );
        }
    }
}

#[test]
#[should_panic(expected = "attempt requires a Ready prepare")]
fn unprepared_attempt_panics() {
    let mut decoder = fixture("overpruning_ladder", config(false));
    let _ = decoder.attempt(decoder.prune_params());
}

#[test]
#[should_panic(expected = "attempt requires a Ready prepare")]
fn fresh_worker_starts_unprepared() {
    let mut decoder = fixture("overpruning_ladder", config(false));
    decoder.prepare(&[0; 4]).unwrap();
    let _ = decoder.fresh_worker().attempt(decoder.prune_params());
}

#[test]
#[should_panic(expected = "attempt requires a Ready prepare")]
fn dimension_error_invalidates_ready() {
    let mut decoder = fixture("overpruning_ladder", config(false));
    decoder.prepare(&[0; 4]).unwrap();
    assert!(decoder.prepare(&[]).is_err());
    let _ = decoder.attempt(decoder.prune_params());
}

#[test]
#[should_panic(expected = "attempt requires a Ready prepare")]
fn residual_invalidates_ready() {
    let mut decoder = fixture("overpruning_ladder", config(false));
    decoder.prepare(&[0; 4]).unwrap();
    assert_eq!(
        decoder.prepare(&[0, 0, 0, 1]).unwrap(),
        TrellisPrepared::Residual { detector: 3 }
    );
    let _ = decoder.attempt(decoder.prune_params());
}

#[test]
fn residual_reports_the_lowest_detector_within_and_across_words() {
    // No probabilistic mechanism touches any detector, so every fired detector
    // beyond the forced D0 is a residual; the report must name the lowest one.
    let mut decoder =
        TrellisDecoder::from_dem_str("error(1) D0 L0\nerror(0) D69", config(false)).unwrap();
    let mut same_word = [0; 70];
    same_word[0] = 1;
    same_word[3] = 1;
    same_word[5] = 1;
    assert_eq!(
        decoder.prepare(&same_word).unwrap(),
        TrellisPrepared::Residual { detector: 3 }
    );
    let mut across_words = [0; 70];
    across_words[0] = 1;
    across_words[5] = 1;
    across_words[69] = 1;
    assert_eq!(
        decoder.prepare(&across_words).unwrap(),
        TrellisPrepared::Residual { detector: 5 }
    );
}

#[test]
fn forced_observables_and_lowest_residual() {
    let mut decoder =
        TrellisDecoder::from_dem_str("error(1) D0 L0 L70\nerror(0) D69", config(false)).unwrap();
    assert_eq!(decoder.forced_observables(), ObsMask::from_words(&[1, 64]));
    assert_eq!(
        decoder.prepare(&[0; 70]).unwrap(),
        TrellisPrepared::Residual { detector: 0 }
    );
    let mut shot = [0; 70];
    shot[0] = 1;
    shot[69] = 1;
    assert_eq!(
        decoder.prepare(&shot).unwrap(),
        TrellisPrepared::Residual { detector: 69 }
    );
    assert_eq!(
        fixture("forced_factors", config(false)).forced_observables(),
        ObsMask::from_u64(2)
    );
    assert!(
        fixture("overpruning_ladder", config(false))
            .forced_observables()
            .is_zero()
    );
}

#[test]
fn residual_precheck_skips_bp_refresh() {
    for integer in [false, true] {
        let mut cfg = config(integer);
        cfg.bp_score_iterations = 5;
        let mut decoder = fixture("overpruning_ladder", cfg);
        assert!(matches!(
            decoder.prepare(&[0; 4]).unwrap(),
            TrellisPrepared::Ready { bp_ran: true, .. }
        ));
        let before = decoder.bp_refreshes();
        assert_eq!(before, 1);
        assert_eq!(
            decoder.prepare(&[0, 0, 0, 1]).unwrap(),
            TrellisPrepared::Residual { detector: 3 }
        );
        assert_eq!(
            decoder.bp_refreshes(),
            before,
            "residual preparation must skip BP"
        );
    }
}
