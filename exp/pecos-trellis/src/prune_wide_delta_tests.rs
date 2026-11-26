//! Differential proof controls for the sufficient wide-Delta guard.
use super::*;

#[test]
fn wide_delta_guard_boundaries_nonfinite_and_rounding_proof() {
    let deltas = [
        1e6_f64.next_down(),
        1e6,
        1e6_f64.next_up(),
        1e9_f64.next_down(),
        1e9,
        1e9_f64.next_up(),
        1e100,
        f64::MAX,
    ];
    assert!(!wide_delta_all_scores_pass(&[], 1e9));
    for delta in [-f64::INFINITY, -1.0, -0.0, 0.0, 1.0, f64::NAN] {
        assert!(!wide_delta_all_scores_pass(&[-0.0, 0.0], delta));
    }
    // Infinite delta admits exactly the finite scores.
    assert!(wide_delta_all_scores_pass(
        &[f64::MIN, -0.0, 0.0, f64::MAX],
        f64::INFINITY
    ));
    for special in [f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
        assert!(!wide_delta_all_scores_pass(&[0.0, special], f64::INFINITY));
    }
    for delta in deltas {
        let bound = delta * 0.25;
        let half = delta * 0.5;
        assert_eq!((bound * 4.0).to_bits(), delta.to_bits());
        assert_eq!((half * 2.0).to_bits(), delta.to_bits());
        let within = [
            -bound,
            (-bound).next_up(),
            -0.0,
            0.0,
            bound.next_down(),
            bound,
        ];
        assert_eq!(wide_delta_all_scores_pass(&within, delta), delta >= 1e6);
        assert!(!wide_delta_all_scores_pass(&[bound.next_up()], delta));
        assert!(!wide_delta_all_scores_pass(&[(-bound).next_down()], delta));
        for special in [
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NAN,
            f64::from_bits(0x7ff0_0000_0000_0001),
            f64::from_bits(0xfff8_1234_5678_9abc),
        ] {
            assert!(!wide_delta_all_scores_pass(&[0.0, special], delta));
        }
        if delta >= 1e6 {
            for scores in [
                within.to_vec(),
                vec![-bound],
                vec![bound],
                vec![-0.0, 0.0],
                vec![-bound, -bound * 0.5],
                vec![bound * 0.5, bound],
            ] {
                assert!(wide_delta_all_scores_pass(&scores, delta));
                let best = scores.iter().copied().max_by(f64::total_cmp).unwrap();
                let cutoff = best - delta;
                assert!(cutoff <= -half);
                assert!(scores.iter().all(|score| *score >= cutoff));
            }
        }
    }
}

/// Prunes through `prune`, which under `cfg(test)` asserts retained keys, mass
/// bits, score bits and telemetry against `prune_reference`. Returns whether
/// the guard took the all-eligible path.
fn assert_same(
    mut actual: FrontierScratch<f64>,
    k: usize,
    delta: f64,
    alpha: f64,
    compatibility: SuffixCompatibility<'_>,
    observed: &[u64],
    mode: DecodeMode,
) -> bool {
    let mut scored = actual.clone();
    score_candidates(&mut scored, alpha, compatibility, observed);
    let guarded = wide_delta_all_scores_pass(&scored.scores, delta);
    prune(&mut actual, k, delta, alpha, compatibility, observed, mode);
    guarded
}

#[test]
fn wide_delta_pruning_matches_reference_all_modes_and_key_orders() {
    let mut exercised = [false; 2];
    for detector_words in [0usize, 2] {
        for logical_words in [1usize, 3] {
            let stride = detector_words + logical_words;
            for count in [1usize, 3, 63, 64, 257] {
                for pattern in 0..6 {
                    let mut frontier = FrontierScratch::<f64> {
                        detector_words,
                        stride,
                        ..Default::default()
                    };
                    for index in 0..count {
                        for word in 0..stride {
                            let value = if word + 1 == stride {
                                index as u64
                            } else {
                                (index as u64)
                                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                                    .rotate_left(
                                        u32::try_from(word * 13).expect("small fixture rotation"),
                                    )
                            };
                            frontier.parent.words.push(value);
                        }
                        frontier.parent.masses.push(match pattern {
                            0 => {
                                -f64::from(
                                    u32::try_from(index * 17 % 29)
                                        .expect("small fixture remainder"),
                                ) / 8.0
                            }
                            1 => {
                                if index % 2 == 0 {
                                    -0.0
                                } else {
                                    0.0
                                }
                            }
                            2 => [
                                -250_000.0,
                                250_000.0,
                                -250_000.0_f64.next_down(),
                                250_000.0_f64.next_up(),
                            ][index % 4],
                            3 => [f64::NEG_INFINITY, f64::INFINITY, -1.0, 0.0][index % 4],
                            // Spread wider than delta at delta=1e6, so the
                            // cutoff drops a state the guard must not admit.
                            4 => [500_000.0, -600_000.0, 0.0, -500_000.0][index % 4],
                            _ => [
                                f64::from_bits(0x7ff8_0000_0000_0123),
                                f64::from_bits(0xfff8_0000_0000_0123),
                                -0.0,
                                0.0,
                                -1.0,
                            ][index % 5],
                        });
                    }
                    frontier.indices.extend((0..count).rev());
                    frontier.retained.resize(count + 7, false);
                    for row_count in [0usize, if detector_words > 0 { 9 } else { 0 }] {
                        let rows: Vec<_> = (0..row_count)
                            .map(|index| SuffixRow {
                                word: 0,
                                bit: u8::try_from(index).expect("nine fixture rows"),
                                epoch: u32::try_from(index).expect("nine fixture rows"),
                            })
                            .collect();
                        let mut values = SuffixValues::new(row_count, None);
                        for (index, pair) in values.probabilities.iter_mut().enumerate() {
                            pair.zero = -0.1
                                * f64::from(u32::try_from(index + 1).expect("nine fixture rows"));
                            pair.one = -1e-9
                                * f64::from(u32::try_from(index + 1).expect("nine fixture rows"))
                                - 3.0;
                        }
                        let compatibility = SuffixCompatibility {
                            incremental: false,
                            rows: &rows,
                            values: &values,
                        };
                        for observation in [0_u64, u64::MAX] {
                            let observed = vec![observation; detector_words];
                            for alpha in [0.0, 0.8] {
                                for k in [0usize, 1, count - 1, count, count + 1, usize::MAX] {
                                    for delta in [
                                        0.0,
                                        1e6_f64.next_down(),
                                        1e6,
                                        1e6_f64.next_up(),
                                        1e9,
                                        f64::MAX,
                                        f64::INFINITY,
                                    ] {
                                        for mode in [DecodeMode::Full, DecodeMode::PredictionOnly] {
                                            let guarded = assert_same(
                                                frontier.clone(),
                                                k,
                                                delta,
                                                alpha,
                                                compatibility,
                                                &observed,
                                                mode,
                                            );
                                            exercised[usize::from(guarded)] = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    assert_eq!(exercised, [true, true]);
}

#[test]
fn wide_delta_extreme_guard_edges_and_empty_contract_match_original() {
    let values = SuffixValues::new(0, None);
    let compatibility = SuffixCompatibility {
        incremental: false,
        rows: &[],
        values: &values,
    };
    for delta in [1e6_f64.next_down(), 1e6, 1e6_f64.next_up(), 1e9, f64::MAX] {
        let bound = delta * 0.25;
        for mass in [
            -bound,
            (-bound).next_up(),
            -0.0,
            0.0,
            bound,
            bound.next_up(),
            (-bound).next_down(),
        ] {
            let mut frontier = FrontierScratch::<f64> {
                stride: 1,
                detector_words: 0,
                ..Default::default()
            };
            frontier.parent.words = vec![9, 1, 7];
            frontier.parent.masses = vec![mass; 3];
            for k in [0usize, 1, 3, usize::MAX] {
                for mode in [DecodeMode::Full, DecodeMode::PredictionOnly] {
                    assert_same(frontier.clone(), k, delta, 0.0, compatibility, &[], mode);
                }
            }
        }
    }
    let empty = FrontierScratch::<f64>::default();
    assert_same(
        empty.clone(),
        usize::MAX,
        f64::INFINITY,
        0.0,
        compatibility,
        &[],
        DecodeMode::Full,
    );
    let a = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prune_impl(
            &mut empty.clone(),
            512,
            1e9,
            0.0,
            compatibility,
            &[],
            DecodeMode::Full,
        );
    }));
    let b = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        prune_reference(
            &mut empty.clone(),
            512,
            1e9,
            0.0,
            compatibility,
            &[],
            DecodeMode::Full,
        );
    }));
    assert!(a.is_err() && b.is_err());
}
