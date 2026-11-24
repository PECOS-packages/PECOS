// Copyright 2026 The PECOS Developers
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "experimental-conditional-retirement")]

use pecos_trellis::{
    ConditionalLogicalRetirement, DecoderError, ObsMask, SparseDem, TrellisConfig,
    TrellisLogicalMass, TrellisResult, TrellisStreamingDecoder,
};
use std::collections::BTreeMap;

fn config(mechanisms: usize) -> TrellisConfig {
    TrellisConfig {
        k: usize::MAX,
        delta: f64::INFINITY,
        score_alpha: 0.0,
        merge_indistinguishable: false,
        dropped_mass: true,
        column_order: Some((0..mechanisms).collect()),
        ..TrellisConfig::default()
    }
}

fn dem(mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>, detectors: usize, logicals: usize) -> SparseDem {
    SparseDem {
        mechanisms,
        detector_coords: BTreeMap::new(),
        num_detectors: detectors,
        num_observables: logicals,
    }
}

fn stream(model: &SparseDem) -> TrellisStreamingDecoder {
    TrellisStreamingDecoder::from_sparse_dem(model, config(model.mechanisms.len())).unwrap()
}

fn mask(bits: &[usize]) -> ObsMask {
    let mut result = ObsMask::new();
    for &bit in bits {
        result.set(bit);
    }
    result
}

fn assert_close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 2e-13,
        "actual={actual:?}, expected={expected:?}"
    );
}

fn assert_invalid(error: &DecoderError) {
    assert!(
        matches!(error, DecoderError::InvalidConfiguration(_)),
        "{error}"
    );
}

fn assert_mass_bits(a: &[TrellisLogicalMass], b: &[TrellisLogicalMass]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.logical, b.logical);
        assert_eq!(a.log_mass.to_bits(), b.log_mass.to_bits());
    }
}

fn assert_result_bits(a: &TrellisResult, b: &TrellisResult) {
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
    assert_eq!(predicted, &b.predicted);
    assert_eq!(log_evidence.to_bits(), b.log_evidence.to_bits());
    assert_eq!(
        runner_up_gap.map(f64::to_bits),
        b.runner_up_gap.map(f64::to_bits)
    );
    assert_eq!(peak_retained_states, &b.peak_retained_states);
    assert_eq!(processed_columns, &b.processed_columns);
    assert_eq!(transitions, &b.transitions);
    assert_eq!(dropped_states, &b.dropped_states);
    assert_eq!(
        dropped_log_mass.map(f64::to_bits),
        b.dropped_log_mass.map(f64::to_bits)
    );
    assert_eq!(bp_seconds.to_bits(), b.bp_seconds.to_bits());
    assert_eq!(bp_runs, &b.bp_runs);
    assert_eq!(escalation_rungs_used, &b.escalation_rungs_used);
    assert_eq!(status, &b.status);
    assert_mass_bits(logical_masses, &b.logical_masses);
}

fn assert_retirement_bits(a: &ConditionalLogicalRetirement, b: &ConditionalLogicalRetirement) {
    assert_eq!(a.selected_class, b.selected_class);
    assert_mass_bits(&a.class_log_masses, &b.class_log_masses);
    assert_eq!(
        a.log_evidence_before.to_bits(),
        b.log_evidence_before.to_bits()
    );
    assert_eq!(
        a.log_evidence_after.to_bits(),
        b.log_evidence_after.to_bits()
    );
    assert_eq!(
        a.runner_up_gap.map(f64::to_bits),
        b.runner_up_gap.map(f64::to_bits)
    );
    assert_eq!(a.states_before, b.states_before);
    assert_eq!(a.states_after, b.states_after);
    assert_eq!(a.removed_states, b.removed_states);
    assert_eq!(a.progress, b.progress);
}

struct ExactFault {
    numerator: u32,
    denominator: u32,
    detectors: &'static [usize],
    logicals: &'static [usize],
}

// Independent exhaustive fault-assignment oracle. Integer products are exact;
// it does not implement trellis transitions, scoring, pruning or retirement.
fn enumerate(faults: &[ExactFault], observed: &[u8]) -> (BTreeMap<u64, u64>, u64) {
    assert!(faults.len() < 16);
    let denominator = faults.iter().map(|f| u64::from(f.denominator)).product();
    let mut classes = BTreeMap::new();
    for assignment in 0..(1_usize << faults.len()) {
        let mut probability = 1_u64;
        let mut detectors = vec![0_u8; observed.len()];
        let mut logicals = 0_u64;
        for (index, fault) in faults.iter().enumerate() {
            assert!(fault.numerator <= fault.denominator);
            let happens = assignment >> index & 1 != 0;
            probability *= u64::from(if happens {
                fault.numerator
            } else {
                fault.denominator - fault.numerator
            });
            if happens {
                for &detector in fault.detectors {
                    detectors[detector] ^= 1;
                }
                for &logical in fault.logicals {
                    assert!(logical < 64);
                    logicals ^= 1_u64 << logical;
                }
            }
        }
        if detectors == observed && probability != 0 {
            *classes.entry(logicals).or_default() += probability;
        }
    }
    (classes, denominator)
}

fn probability(numerator: u64, denominator: u64) -> f64 {
    f64::from(u32::try_from(numerator).unwrap()) / f64::from(u32::try_from(denominator).unwrap())
}

#[test]
fn marginal_map_beats_best_state_and_stable_filter_preserves_mass_bits() {
    let model = dem(
        vec![
            (0.85, vec![0], vec![0]),
            (0.1, vec![0], vec![1]),
            (0.1, vec![0], vec![2]),
        ],
        1,
        3,
    );
    let (oracle, denominator) = enumerate(
        &[
            ExactFault {
                numerator: 85,
                denominator: 100,
                detectors: &[0],
                logicals: &[0],
            },
            ExactFault {
                numerator: 1,
                denominator: 10,
                detectors: &[0],
                logicals: &[1],
            },
            ExactFault {
                numerator: 1,
                denominator: 10,
                detectors: &[0],
                logicals: &[2],
            },
        ],
        &[0],
    );
    let best_state = oracle.iter().max_by_key(|(_, weight)| *weight).unwrap();
    assert_eq!(
        *best_state.0, 0,
        "largest individual state is outside winning tag1 class"
    );
    let tag1: u64 = oracle
        .iter()
        .filter(|(key, _)| **key & 1 != 0)
        .map(|(_, weight)| *weight)
        .sum();
    let tag0: u64 = oracle
        .iter()
        .filter(|(key, _)| **key & 1 == 0)
        .map(|(_, weight)| *weight)
        .sum();
    assert!(tag1 > tag0);

    let mut snapshot = stream(&model);
    snapshot.feed_dense(&[0]).unwrap();
    snapshot.advance_prediction().unwrap();
    // At terminal every detector is closed. Requesting all logical bits exposes
    // each complete original retained key/mass before that control is filtered.
    let all_classes = snapshot.retire_completed_logical_class(&[0, 1, 2]).unwrap();
    let expected_survivors: Vec<_> = all_classes
        .class_log_masses
        .into_iter()
        .filter(|m| m.logical.get(0))
        .collect();

    let mut subject = stream(&model);
    subject.feed_dense(&[0]).unwrap();
    subject.advance_prediction().unwrap();
    assert!(!subject.committed().1.get(0));
    let retired = subject.retire_completed_logical_class(&[0]).unwrap();
    assert_eq!(retired.selected_class, mask(&[0]));
    assert_eq!(
        (
            retired.states_before,
            retired.states_after,
            retired.removed_states
        ),
        (4, 2, 2)
    );
    assert_eq!(retired.class_log_masses.len(), 2);
    assert_eq!(retired.class_log_masses[0].logical, mask(&[0]));
    assert_close(
        retired.class_log_masses[0].log_mass.exp(),
        probability(tag1, denominator),
    );
    assert_close(
        retired.class_log_masses[1].log_mass.exp(),
        probability(tag0, denominator),
    );
    assert_close(
        retired.log_evidence_before.exp(),
        probability(tag0 + tag1, denominator),
    );
    assert_eq!(
        retired.log_evidence_after.to_bits(),
        retired.class_log_masses[0].log_mass.to_bits()
    );
    assert_close(
        retired.runner_up_gap.unwrap(),
        (probability(tag1, tag0)).ln(),
    );
    assert!(retired.progress.committed.get(0) && retired.progress.committed_mask.get(0));
    let result = subject.flush().unwrap();
    assert_mass_bits(&result.logical_masses, &expected_survivors);
    assert_eq!(
        result.dropped_states, 0,
        "conditioning is not ordinary beam pruning"
    );
    assert_close(result.log_evidence.exp(), probability(tag1, denominator));
    let repeated = subject.retire_completed_logical_class(&[0]).unwrap();
    assert_eq!(repeated.class_log_masses.len(), 1);
    assert_eq!(repeated.states_before, repeated.states_after);
    assert_eq!(repeated.removed_states, 0);
    assert_eq!(
        repeated.log_evidence_before.to_bits(),
        repeated.log_evidence_after.to_bits()
    );
    assert_eq!(repeated.progress.newly_committed, [] as [(usize, bool); 0]);
}

#[test]
fn later_evidence_can_regret_a_completed_tag_without_beam_pruning() {
    // Core API checks tag completion, not ownership/horizon closure. D0 is
    // still open after the first column; D1 later identifies its best cause.
    let model = dem(
        vec![
            (0.4, vec![0], vec![0, 1]),
            (0.1, vec![0, 1], vec![]),
            (0.01, vec![1], vec![]),
        ],
        2,
        2,
    );
    let (oracle, denominator) = enumerate(
        &[
            ExactFault {
                numerator: 2,
                denominator: 5,
                detectors: &[0],
                logicals: &[0, 1],
            },
            ExactFault {
                numerator: 1,
                denominator: 10,
                detectors: &[0, 1],
                logicals: &[],
            },
            ExactFault {
                numerator: 1,
                denominator: 100,
                detectors: &[1],
                logicals: &[],
            },
        ],
        &[1, 0],
    );
    assert!(oracle[&3] > oracle[&0]);
    let mut unconditioned = stream(&model);
    unconditioned.feed_dense(&[1, 0]).unwrap();
    let baseline = unconditioned.flush().unwrap();
    assert!(baseline.predicted.get(1));
    assert_eq!(baseline.dropped_states, 0);

    let mut conditioned = stream(&model);
    conditioned.feed_prefix(&[1]).unwrap();
    assert_eq!(conditioned.advance().unwrap().columns_processed, 1);
    let retired = conditioned.retire_completed_logical_class(&[0]).unwrap();
    assert_eq!(retired.selected_class, ObsMask::new());
    assert_close(retired.log_evidence_before.exp(), 1.0);
    assert_close(retired.log_evidence_after.exp(), 0.6);
    assert!(retired.progress.committed_mask.get(0));
    assert!(!retired.progress.committed.get(0));
    conditioned.feed_prefix(&[0]).unwrap();
    let result = conditioned.flush().unwrap();
    assert!(
        !result.predicted.get(1),
        "the selected past tag is not recovered"
    );
    assert_close(
        result.log_evidence.exp(),
        probability(oracle[&0], denominator),
    );
    assert_eq!(result.dropped_states, 0);
}

#[test]
fn regret_can_remove_every_future_path_and_failure_precedes_bad_requests_until_reset() {
    let model = dem(
        vec![(0.1, vec![0], vec![0]), (0.01, vec![0, 1], vec![])],
        2,
        1,
    );
    let mut subject = stream(&model);
    subject.feed_prefix(&[1]).unwrap();
    assert_eq!(subject.advance().unwrap().columns_processed, 1);
    subject.retire_completed_logical_class(&[0]).unwrap();
    subject.feed_prefix(&[0]).unwrap();
    let failure = subject.advance().unwrap_err();
    assert!(matches!(failure, DecoderError::DecodingFailed(_)));
    for request in [vec![], vec![0, 0], vec![usize::MAX]] {
        let actual = subject
            .retire_completed_logical_class(&request)
            .unwrap_err();
        assert_eq!(
            actual.to_string(),
            failure.to_string(),
            "sticky failure must outrank invalid request"
        );
    }
    subject.reset();
    subject.feed_dense(&[1, 0]).unwrap();
    let baseline = subject.flush().unwrap();
    assert!(
        baseline.predicted.get(0),
        "reset removes the regretful condition"
    );
    let result = subject.retire_completed_logical_class(&[0]).unwrap();
    assert!(result.selected_class.get(0));
}

#[test]
fn malformed_requests_leave_frontier_pending_progress_and_full_result_unchanged() {
    // Padding L2 has a construction-time pending zero commitment. An invalid
    // retirement must not consume it while examining the stochastic L0/L1.
    let model = dem(vec![(0.2, vec![], vec![0]), (0.3, vec![], vec![1])], 0, 3);
    for request in [vec![], vec![0, 0], vec![3], vec![1, 0, 1]] {
        let mut subject = stream(&model);
        let mut control = stream(&model);
        subject.advance_prediction().unwrap();
        control.advance_prediction().unwrap();
        let before = subject.committed();
        assert_invalid(
            &subject
                .retire_completed_logical_class(&request)
                .unwrap_err(),
        );
        assert_eq!(subject.committed(), before);
        let actual_progress = subject.advance().unwrap();
        let expected_progress = control.advance().unwrap();
        assert!(expected_progress.newly_committed.contains(&(2, false)));
        assert_eq!(
            actual_progress, expected_progress,
            "invalid request cannot consume pending progress"
        );
        assert_result_bits(&subject.flush().unwrap(), &control.flush().unwrap());
    }
}

#[test]
fn unprocessed_last_toggle_is_rejected_without_changing_future_behavior() {
    let model = dem(vec![(0.2, vec![0], vec![0]), (0.3, vec![1], vec![0])], 2, 1);
    let mut subject = stream(&model);
    let mut control = stream(&model);
    subject.feed_prefix(&[1]).unwrap();
    control.feed_prefix(&[1]).unwrap();
    assert_eq!(subject.advance().unwrap(), control.advance().unwrap());
    let before = subject.committed();
    assert_invalid(&subject.retire_completed_logical_class(&[0]).unwrap_err());
    assert_eq!(subject.committed(), before);
    subject.feed_prefix(&[0]).unwrap();
    control.feed_prefix(&[0]).unwrap();
    assert_result_bits(&subject.flush().unwrap(), &control.flush().unwrap());
}

#[test]
fn pending_bounded_phase_even_drained_blocks_retirement_without_consuming_it() {
    let model = dem(vec![(0.2, vec![], vec![0])], 0, 1);
    let mut subject = stream(&model);
    assert_eq!(subject.begin_bounded_advance().unwrap(), 1);
    assert_invalid(&subject.retire_completed_logical_class(&[0]).unwrap_err());
    assert_eq!(subject.bounded_columns_remaining(), Some(1));
    assert!(subject.step_bounded_advance(1).unwrap().1);
    let before = subject.committed();
    assert_invalid(&subject.retire_completed_logical_class(&[0]).unwrap_err());
    assert_eq!(subject.bounded_columns_remaining(), Some(0));
    assert_eq!(subject.committed(), before);
    subject.finish_bounded_advance().unwrap();
    let retired = subject.retire_completed_logical_class(&[0]).unwrap();
    assert_eq!(retired.states_before, 2);
    assert_eq!(retired.states_after, 1);
}

#[test]
fn forced_probability_one_zero_and_never_toggled_bits_are_closed_without_column_work() {
    let model = dem(
        vec![
            (1.0, vec![], vec![2]),
            (0.0, vec![], vec![1]),
            (0.3, vec![], vec![0]),
        ],
        0,
        4,
    );
    let mut subject = stream(&model);
    let retired = subject.retire_completed_logical_class(&[3, 2, 1]).unwrap();
    assert_eq!(retired.selected_class, mask(&[2]));
    assert_eq!(retired.progress.columns_processed, 0);
    assert_eq!((retired.states_before, retired.states_after), (1, 1));
    assert_eq!(retired.class_log_masses.len(), 1);
    assert_eq!(retired.log_evidence_before.to_bits(), 0.0_f64.to_bits());
    assert_eq!(retired.runner_up_gap, None);
    assert_invalid(&subject.retire_completed_logical_class(&[0]).unwrap_err());
    subject.advance().unwrap();
    let random = subject.retire_completed_logical_class(&[0]).unwrap();
    assert_eq!(random.selected_class, ObsMask::new());
    let result = subject.flush().unwrap();
    assert!(
        result.predicted.get(2),
        "conditioning never clears the unrelated forced bit"
    );
    assert!(!result.predicted.get(0) && !result.predicted.get(1) && !result.predicted.get(3));
    assert_close(result.log_evidence.exp(), 0.7);
}

#[test]
fn equal_mass_classes_use_unsigned_wide_mask_order_and_keep_original_bit_positions() {
    let model = dem(
        vec![
            (0.5, vec![], vec![129]),
            (1.0, vec![], vec![65]),
            (0.5, vec![], vec![70]),
        ],
        0,
        130,
    );
    let mut subject = stream(&model);
    subject.advance().unwrap();
    let retired = subject.retire_completed_logical_class(&[129, 70]).unwrap();
    let expected = [mask(&[]), mask(&[70]), mask(&[129]), mask(&[70, 129])];
    assert_eq!(
        retired
            .class_log_masses
            .iter()
            .map(|class| class.logical.clone())
            .collect::<Vec<_>>(),
        expected
    );
    for class in &retired.class_log_masses {
        assert_eq!(
            class.log_mass.to_bits(),
            retired.class_log_masses[0].log_mass.to_bits()
        );
        assert!(!class.logical.get(65));
    }
    assert_eq!(retired.selected_class, ObsMask::new());
    assert_eq!(retired.runner_up_gap.unwrap().to_bits(), 0.0_f64.to_bits());
    assert_eq!((retired.states_before, retired.states_after), (4, 1));
    let result = subject.flush().unwrap();
    assert_eq!(result.predicted, mask(&[65]));
    assert_close(result.log_evidence.exp(), 0.25);
}

#[test]
fn conditioning_does_not_increment_existing_beam_drop_counters_or_rewrite_status() {
    let model = dem(vec![(0.4, vec![], vec![0]), (0.3, vec![], vec![1])], 0, 2);
    let cfg = TrellisConfig {
        k: 3,
        ..config(model.mechanisms.len())
    };
    let mut control = TrellisStreamingDecoder::from_sparse_dem(&model, cfg.clone()).unwrap();
    let baseline = control.flush().unwrap();
    assert!(baseline.dropped_states > 0);
    let mut subject = TrellisStreamingDecoder::from_sparse_dem(&model, cfg).unwrap();
    subject.advance().unwrap();
    let retired = subject.retire_completed_logical_class(&[0]).unwrap();
    assert!(retired.removed_states > 0);
    let result = subject.flush().unwrap();
    assert_eq!(result.dropped_states, baseline.dropped_states);
    assert_eq!(
        result.dropped_log_mass.map(f64::to_bits),
        baseline.dropped_log_mass.map(f64::to_bits)
    );
    assert_eq!(result.status, baseline.status);
    assert_eq!(result.transitions, baseline.transitions);
    assert_eq!(result.processed_columns, baseline.processed_columns);
    assert_eq!(result.peak_retained_states, baseline.peak_retained_states);
}

#[test]
fn bounded_and_unbounded_retirement_match_every_mass_bit_for_all_syndromes_and_budgets() {
    let model = dem(
        vec![
            (0.2, vec![0], vec![0]),
            (0.3, vec![0], vec![1]),
            (0.4, vec![1], vec![1]),
            (0.25, vec![1], vec![2]),
            (0.15, vec![2], vec![2]),
            (0.1, vec![2], vec![0]),
        ],
        3,
        3,
    );
    for budget in [1, 2, 32] {
        for bits in 0..8_u8 {
            let mut bounded = stream(&model);
            let mut plain = stream(&model);
            for block in 0..3 {
                let value = [bits >> block & 1];
                bounded.feed_prefix(&value).unwrap();
                plain.feed_prefix(&value).unwrap();
                bounded.begin_bounded_advance().unwrap();
                while bounded.bounded_columns_remaining().unwrap() != 0 {
                    bounded.step_bounded_advance(budget).unwrap();
                }
                assert_eq!(
                    bounded.finish_bounded_advance().unwrap(),
                    plain.advance().unwrap()
                );
                let requested: &[usize] = match block {
                    1 => &[1],
                    2 => &[0, 2],
                    _ => continue,
                };
                let a = bounded.retire_completed_logical_class(requested).unwrap();
                let b = plain.retire_completed_logical_class(requested).unwrap();
                assert_retirement_bits(&a, &b);
            }
            assert_result_bits(&bounded.flush().unwrap(), &plain.flush().unwrap());
        }
    }
}
