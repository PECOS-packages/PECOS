// Copyright 2026 The PECOS Developers
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "experimental-conditional-retirement")]

use pecos_trellis::{
    ConditionalLogicalClassPreview, ConditionalLogicalRetirement,
    ConditionalLogicalRetirementDecision, DecoderError, ObsMask, SparseDem, TrellisConfig,
    TrellisLogicalMass, TrellisResult, TrellisStreamingDecoder,
};
use std::collections::BTreeMap;

fn model(
    mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>,
    detectors: usize,
    logicals: usize,
) -> SparseDem {
    SparseDem {
        mechanisms,
        detector_coords: BTreeMap::new(),
        num_detectors: detectors,
        num_observables: logicals,
    }
}

fn stream(dem: &SparseDem) -> TrellisStreamingDecoder {
    TrellisStreamingDecoder::from_sparse_dem(
        dem,
        TrellisConfig {
            k: usize::MAX,
            delta: f64::INFINITY,
            score_alpha: 0.0,
            merge_indistinguishable: false,
            dropped_mass: true,
            column_order: Some((0..dem.mechanisms.len()).collect()),
            ..TrellisConfig::default()
        },
    )
    .unwrap()
}

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() < 3e-13,
        "{actual:?} != {expected:?}"
    );
}

fn invalid(error: &DecoderError) {
    assert!(
        matches!(error, DecoderError::InvalidConfiguration(_)),
        "{error}"
    );
}

fn mass_bits(a: &[TrellisLogicalMass], b: &[TrellisLogicalMass]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(a.logical, b.logical);
        assert_eq!(a.log_mass.to_bits(), b.log_mass.to_bits());
    }
}

fn result_bits(a: &TrellisResult, b: &TrellisResult) {
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
    mass_bits(logical_masses, &b.logical_masses);
}

fn preview_bits(a: &ConditionalLogicalClassPreview, b: &ConditionalLogicalClassPreview) {
    assert_eq!(a.selected_class, b.selected_class);
    mass_bits(&a.class_log_masses, &b.class_log_masses);
    assert_eq!(
        a.log_evidence_before.to_bits(),
        b.log_evidence_before.to_bits()
    );
    assert_eq!(a.selected_log_mass.to_bits(), b.selected_log_mass.to_bits());
    assert_eq!(
        a.runner_up_gap.map(f64::to_bits),
        b.runner_up_gap.map(f64::to_bits)
    );
    assert_eq!(a.states_before, b.states_before);
    assert_eq!(a.selected_state_count, b.selected_state_count);
}

fn retirement_bits(a: &ConditionalLogicalRetirement, b: &ConditionalLogicalRetirement) {
    assert_eq!(a.selected_class, b.selected_class);
    mass_bits(&a.class_log_masses, &b.class_log_masses);
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

fn decision_bits(
    a: &ConditionalLogicalRetirementDecision,
    b: &ConditionalLogicalRetirementDecision,
) {
    match (a, b) {
        (
            ConditionalLogicalRetirementDecision::Deferred(a),
            ConditionalLogicalRetirementDecision::Deferred(b),
        ) => preview_bits(a, b),
        (
            ConditionalLogicalRetirementDecision::Retired(a),
            ConditionalLogicalRetirementDecision::Retired(b),
        ) => retirement_bits(a, b),
        _ => panic!("different decisions: {a:?} / {b:?}"),
    }
}

fn deferred(decision: ConditionalLogicalRetirementDecision) -> ConditionalLogicalClassPreview {
    match decision {
        ConditionalLogicalRetirementDecision::Deferred(preview) => preview,
        other @ ConditionalLogicalRetirementDecision::Retired(_) => {
            panic!("expected Deferred: {other:?}")
        }
    }
}

fn retired(decision: ConditionalLogicalRetirementDecision) -> ConditionalLogicalRetirement {
    match decision {
        ConditionalLogicalRetirementDecision::Retired(result) => result,
        other @ ConditionalLogicalRetirementDecision::Deferred(_) => {
            panic!("expected Retired: {other:?}")
        }
    }
}

#[test]
fn preview_and_deferred_leave_pending_progress_state_and_every_future_mass_bit_unchanged() {
    // Padding L2 has an unconsumed construction-time commitment. A read-only
    // preview/Deferred must not call take_progress or condition the other bits.
    let dem = model(vec![(0.2, vec![], vec![0]), (0.3, vec![], vec![1])], 0, 3);
    let mut subject = stream(&dem);
    let mut control = stream(&dem);
    subject.advance_prediction().unwrap();
    control.advance_prediction().unwrap();
    let before = format!("{subject:?}");
    let first = subject.preview_completed_logical_class(&[0]).unwrap();
    assert_eq!(first.states_before, 4);
    assert_eq!(first.selected_state_count, 2);
    close(first.log_evidence_before.exp(), 1.0);
    close(first.selected_log_mass.exp(), 0.8);
    for _ in 0..3 {
        preview_bits(
            &first,
            &subject.preview_completed_logical_class(&[0]).unwrap(),
        );
        let decision = subject
            .try_retire_completed_logical_class(&[0], 0.9)
            .unwrap();
        preview_bits(&first, &deferred(decision));
        assert_eq!(format!("{subject:?}"), before);
    }
    let expected_progress = control.advance().unwrap();
    assert!(expected_progress.newly_committed.contains(&(2, false)));
    assert_eq!(subject.advance().unwrap(), expected_progress);
    result_bits(&subject.flush().unwrap(), &control.flush().unwrap());
}

#[test]
fn probability_zero_is_atomic_legacy_retirement_in_every_returned_field() {
    // Each observed parity has multiple full keys per masked class. This also
    // carries an unrelated pending L2 commitment into the retirement result.
    let dem = model(
        vec![
            (0.22, vec![0], vec![0, 1]),
            (0.2, vec![0], vec![0]),
            (0.1, vec![0], vec![]),
        ],
        1,
        3,
    );
    for syndrome in [0, 1] {
        for logicals in [vec![0], vec![1], vec![0, 1]] {
            let mut subject = stream(&dem);
            let mut legacy = stream(&dem);
            subject.feed_dense(&[syndrome]).unwrap();
            legacy.feed_dense(&[syndrome]).unwrap();
            subject.advance_prediction().unwrap();
            legacy.advance_prediction().unwrap();
            let expected = legacy.retire_completed_logical_class(&logicals).unwrap();
            let actual = retired(
                subject
                    .try_retire_completed_logical_class(&logicals, 0.0)
                    .unwrap(),
            );
            retirement_bits(&actual, &expected);
            result_bits(&subject.flush().unwrap(), &legacy.flush().unwrap());
        }
    }
}

#[test]
fn ninety_percent_uses_normalized_base_class_mass_and_joint_correlations() {
    eprintln!("V5_LIBM_LOG_P09_BITS=0x{:016x}", libm::log(0.9).to_bits());
    // Independent fault products give L0=0 mass 89/100 or 91/100. Splitting
    // that class over L1 makes its largest individual state far below 0.9.
    for (p, probability, should_retire) in [(0.11, 0.89, false), (0.09, 0.91, true)] {
        let dem = model(vec![(p, vec![], vec![0]), (0.4, vec![], vec![1])], 0, 2);
        let mut subject = stream(&dem);
        subject.advance_prediction().unwrap();
        let preview = subject.preview_completed_logical_class(&[0]).unwrap();
        close(
            (preview.selected_log_mass - preview.log_evidence_before).exp(),
            probability,
        );
        assert_eq!(preview.selected_state_count, 2);
        let decision = subject
            .try_retire_completed_logical_class(&[0], 0.9)
            .unwrap();
        assert_eq!(
            matches!(decision, ConditionalLogicalRetirementDecision::Retired(_)),
            should_retire
        );
    }
    // Independent A=.02 toggles both bits; B=C=.06 toggle one each.
    // P00=.98*.94*.94+.02*.06*.06=.866, while either marginal P0=.9224.
    // Both individually exceed 0.9; the requested JOINT class does not.
    let dem = model(
        vec![
            (0.02, vec![], vec![0, 1]),
            (0.06, vec![], vec![0]),
            (0.06, vec![], vec![1]),
        ],
        0,
        2,
    );
    let mut subject = stream(&dem);
    subject.advance_prediction().unwrap();
    for logical in [0, 1] {
        let p = subject.preview_completed_logical_class(&[logical]).unwrap();
        close((p.selected_log_mass - p.log_evidence_before).exp(), 0.9224);
    }
    let p = deferred(
        subject
            .try_retire_completed_logical_class(&[0, 1], 0.9)
            .unwrap(),
    );
    close((p.selected_log_mass - p.log_evidence_before).exp(), 0.866);
    assert_eq!(p.class_log_masses.len(), 4);
}

#[test]
fn probability_one_requires_single_class_even_if_exp_rounds_to_one() {
    let dem = model(vec![(1e-30, vec![], vec![0])], 0, 1);
    let mut subject = stream(&dem);
    subject.advance_prediction().unwrap();
    let p = subject.preview_completed_logical_class(&[0]).unwrap();
    assert_eq!(p.class_log_masses.len(), 2);
    assert_eq!(
        (p.selected_log_mass - p.log_evidence_before)
            .exp()
            .to_bits(),
        1.0_f64.to_bits()
    );
    let before = format!("{subject:?}");
    deferred(
        subject
            .try_retire_completed_logical_class(&[0], 1.0)
            .unwrap(),
    );
    assert_eq!(format!("{subject:?}"), before);
    let chosen = retired(
        subject
            .try_retire_completed_logical_class(&[0], 0.99)
            .unwrap(),
    );
    assert_eq!(chosen.states_after, 1);
    let single = retired(
        subject
            .try_retire_completed_logical_class(&[0], 1.0)
            .unwrap(),
    );
    assert_eq!(single.class_log_masses.len(), 1);
    assert_eq!(single.removed_states, 0);
}

#[test]
fn later_prefix_can_change_a_deferred_completed_class() {
    let dem = model(
        vec![(0.2, vec![0], vec![0]), (0.01, vec![0, 1], vec![])],
        2,
        1,
    );
    let mut subject = stream(&dem);
    subject.feed_prefix(&[1]).unwrap();
    assert_eq!(subject.advance().unwrap().columns_processed, 1);
    let first = deferred(
        subject
            .try_retire_completed_logical_class(&[0], 0.9)
            .unwrap(),
    );
    assert!(!first.selected_class.get(0));
    close(
        (first.selected_log_mass - first.log_evidence_before).exp(),
        0.8,
    );
    subject.feed_prefix(&[0]).unwrap();
    subject.advance().unwrap();
    let later = retired(
        subject
            .try_retire_completed_logical_class(&[0], 0.9)
            .unwrap(),
    );
    assert!(later.selected_class.get(0));
    let mut control = stream(&dem);
    control.feed_dense(&[1, 0]).unwrap();
    result_bits(&subject.flush().unwrap(), &control.flush().unwrap());
}

#[test]
fn forced_zero_one_and_never_toggled_bits_allow_single_class_without_work() {
    let dem = model(
        vec![
            (1.0, vec![], vec![2]),
            (0.0, vec![], vec![1]),
            (0.3, vec![], vec![0]),
        ],
        0,
        4,
    );
    let mut subject = stream(&dem);
    let before = format!("{subject:?}");
    let p = subject.preview_completed_logical_class(&[3, 2, 1]).unwrap();
    assert_eq!(format!("{subject:?}"), before);
    assert_eq!(p.class_log_masses.len(), 1);
    assert!(p.selected_class.get(2));
    assert!(!p.selected_class.get(1) && !p.selected_class.get(3));
    let result = retired(
        subject
            .try_retire_completed_logical_class(&[3, 2, 1], 1.0)
            .unwrap(),
    );
    assert_eq!(result.progress.columns_processed, 0);
    assert_eq!(
        (
            result.states_before,
            result.states_after,
            result.removed_states
        ),
        (1, 1, 0)
    );
    invalid(&subject.preview_completed_logical_class(&[0]).unwrap_err());
}

#[test]
fn wide_mask_preview_keeps_original_positions_and_unsigned_ties() {
    let dem = model(
        vec![(0.5, vec![], vec![63, 64]), (1.0, vec![], vec![64])],
        0,
        130,
    );
    let mut subject = stream(&dem);
    subject.advance_prediction().unwrap();
    let p = deferred(
        subject
            .try_retire_completed_logical_class(&[129, 64, 63], 0.9)
            .unwrap(),
    );
    assert!(p.selected_class.get(63));
    assert!(!p.selected_class.get(64) && !p.selected_class.get(129));
    assert_eq!(p.runner_up_gap.unwrap().to_bits(), 0.0_f64.to_bits());
    assert_eq!(p.class_log_masses.len(), 2);
    let value = retired(
        subject
            .try_retire_completed_logical_class(&[129, 64, 63], 0.0)
            .unwrap(),
    );
    assert_eq!(value.selected_class, ObsMask::from_words(&[1_u64 << 63]));
}

#[test]
fn invalid_requests_thresholds_and_bounded_phases_are_non_consuming() {
    let dem = model(vec![(0.2, vec![], vec![0]), (0.3, vec![], vec![1])], 0, 3);
    let mut subject = stream(&dem);
    subject.advance_prediction().unwrap();
    for request in [vec![], vec![0, 0], vec![3], vec![1, 0, 1]] {
        let before = format!("{subject:?}");
        invalid(
            &subject
                .preview_completed_logical_class(&request)
                .unwrap_err(),
        );
        invalid(
            &subject
                .try_retire_completed_logical_class(&request, 0.9)
                .unwrap_err(),
        );
        assert_eq!(format!("{subject:?}"), before);
    }
    for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
        let before = format!("{subject:?}");
        invalid(
            &subject
                .try_retire_completed_logical_class(&[0], threshold)
                .unwrap_err(),
        );
        assert_eq!(format!("{subject:?}"), before);
    }
    let pending_model = model(vec![(0.2, vec![], vec![0])], 0, 1);
    let mut pending = stream(&pending_model);
    assert_eq!(pending.begin_bounded_advance().unwrap(), 1);
    for drained in [false, true] {
        if drained {
            assert!(pending.step_bounded_advance(1).unwrap().1);
        }
        let before = format!("{pending:?}");
        invalid(&pending.preview_completed_logical_class(&[0]).unwrap_err());
        invalid(
            &pending
                .try_retire_completed_logical_class(&[0], 0.9)
                .unwrap_err(),
        );
        assert_eq!(format!("{pending:?}"), before);
    }
    pending.finish_bounded_advance().unwrap();
    deferred(
        pending
            .try_retire_completed_logical_class(&[0], 0.9)
            .unwrap(),
    );
}

#[test]
fn sticky_failure_precedes_bad_preview_and_threshold_then_reset_restores_state() {
    let dem = model(
        vec![(0.1, vec![0], vec![0]), (0.01, vec![0, 1], vec![])],
        2,
        1,
    );
    let mut subject = stream(&dem);
    subject.feed_prefix(&[1]).unwrap();
    subject.advance().unwrap();
    retired(
        subject
            .try_retire_completed_logical_class(&[0], 0.0)
            .unwrap(),
    );
    subject.feed_prefix(&[0]).unwrap();
    let failure = subject.advance().unwrap_err();
    assert!(matches!(failure, DecoderError::DecodingFailed(_)));
    assert_eq!(
        subject
            .preview_completed_logical_class(&[])
            .unwrap_err()
            .to_string(),
        failure.to_string()
    );
    assert_eq!(
        subject
            .try_retire_completed_logical_class(&[], f64::NAN)
            .unwrap_err()
            .to_string(),
        failure.to_string()
    );
    subject.reset();
    subject.feed_dense(&[1, 0]).unwrap();
    subject.advance().unwrap();
    let result = retired(
        subject
            .try_retire_completed_logical_class(&[0], 1.0)
            .unwrap(),
    );
    assert_eq!(result.class_log_masses.len(), 1);
    let mut fresh = stream(&dem);
    fresh.feed_dense(&[1, 0]).unwrap();
    fresh.advance().unwrap();
    fresh.retire_completed_logical_class(&[0]).unwrap();
    result_bits(&subject.flush().unwrap(), &fresh.flush().unwrap());
}

#[test]
fn bounded_and_unbounded_decisions_match_every_mass_bit_for_all_toy_syndromes() {
    let dem = model(
        vec![
            (0.08, vec![0], vec![0]),
            (0.2, vec![0], vec![]),
            (0.12, vec![1], vec![1]),
            (0.3, vec![1], vec![]),
            (0.18, vec![2], vec![2]),
            (0.1, vec![2], vec![]),
        ],
        3,
        3,
    );
    for syndrome in 0_u8..8 {
        for budget in [1, 3, 32] {
            let mut subject = stream(&dem);
            let mut control = stream(&dem);
            for row in 0..3 {
                let bit = (syndrome >> row) & 1;
                subject.feed_prefix(&[bit]).unwrap();
                control.feed_prefix(&[bit]).unwrap();
                subject.begin_bounded_advance().unwrap();
                while subject.bounded_columns_remaining().unwrap() != 0 {
                    subject.step_bounded_advance(budget).unwrap();
                }
                assert_eq!(
                    subject.finish_bounded_advance().unwrap(),
                    control.advance().unwrap()
                );
                let logicals: Vec<_> = (0..=row).collect();
                preview_bits(
                    &subject.preview_completed_logical_class(&logicals).unwrap(),
                    &control.preview_completed_logical_class(&logicals).unwrap(),
                );
                let actual = subject
                    .try_retire_completed_logical_class(&logicals, 0.9)
                    .unwrap();
                let expected = control
                    .try_retire_completed_logical_class(&logicals, 0.9)
                    .unwrap();
                decision_bits(&actual, &expected);
            }
            result_bits(&subject.flush().unwrap(), &control.flush().unwrap());
        }
    }
}
