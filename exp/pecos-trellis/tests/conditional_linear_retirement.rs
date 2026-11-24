// Copyright 2026 The PECOS Developers
// SPDX-License-Identifier: Apache-2.0

#![cfg(feature = "experimental-conditional-retirement")]

use pecos_trellis::{
    DecoderError, ObsMask, SparseDem, TrellisConfig, TrellisLogicalMass, TrellisResult,
    TrellisStreamingDecoder,
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

fn mask(bits: &[usize]) -> ObsMask {
    let mut value = ObsMask::new();
    for &bit in bits {
        value.set(bit);
    }
    value
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

#[test]
fn xor_class_sums_over_full_tuples_and_preserves_every_surviving_mass_bit() {
    // The four D0=1 assignments have exact probability products:
    // A=.1584 -> L1, B=.1404 -> L0+L1, C=.0624 -> 0,
    // ABC=.0044 -> L0. Thus best full tuple A has XOR=1, but
    // XOR=0 wins after aggregation: .1404+.0624 > .1584+.0044.
    let dem = model(
        vec![
            (0.22, vec![0], vec![1]),
            (0.2, vec![0], vec![0, 1]),
            (0.1, vec![0], vec![]),
        ],
        1,
        2,
    );
    let mut baseline = stream(&dem);
    baseline.feed_prefix(&[1]).unwrap();
    let unconditioned = baseline.flush().unwrap();
    assert_eq!(unconditioned.predicted, mask(&[1]));
    let mut subject = stream(&dem);
    subject.feed_prefix(&[1]).unwrap();
    subject.advance().unwrap();
    let projection = [mask(&[0, 1])];
    let preview = subject.preview_completed_linear_class(&projection).unwrap();
    assert_eq!(preview.selected_class, mask(&[]));
    assert_eq!(preview.states_before, 4);
    assert_eq!(preview.selected_state_count, 2);
    close(preview.log_evidence_before.exp(), 0.3656);
    close(preview.selected_log_mass.exp(), 0.2028);
    assert_eq!(preview.class_log_masses[1].logical, mask(&[0]));
    close(preview.class_log_masses[1].log_mass.exp(), 0.1628);
    let retired = subject.retire_completed_linear_class(&projection).unwrap();
    mass_bits(&preview.class_log_masses, &retired.class_log_masses);
    assert_eq!(
        retired.log_evidence_before.to_bits(),
        preview.log_evidence_before.to_bits()
    );
    assert_eq!(
        retired.log_evidence_after.to_bits(),
        preview.selected_log_mass.to_bits()
    );
    assert_eq!(
        (
            retired.states_before,
            retired.states_after,
            retired.removed_states
        ),
        (4, 2, 2)
    );
    let conditional = subject.flush().unwrap();
    let expected: Vec<_> = unconditioned
        .logical_masses
        .iter()
        .filter(|class| class.logical.get(0) == class.logical.get(1))
        .cloned()
        .collect();
    mass_bits(&conditional.logical_masses, &expected);
    assert_eq!(conditional.predicted, mask(&[0, 1]));
    assert_eq!(conditional.dropped_states, unconditioned.dropped_states);
    assert_eq!(conditional.transitions, unconditioned.transitions);
    close(conditional.log_evidence.exp(), 0.2028);
}

#[test]
fn winning_projection_retains_boundary_and_physical_logical_correlations() {
    // D1 is known but remains an open boundary when XOR(L0,L1) retires.
    // The final unlabelled fault E touches D1 and still-unknown D2, so the
    // first three columns are ready and E is not. Later D2 fixes E; keeping
    // the D1/full-key correlation must then select B or C as shown below.
    let dem = model(
        vec![
            (0.22, vec![0], vec![1]),
            (0.2, vec![0, 1], vec![0, 1]),
            (0.1, vec![0], vec![]),
            (0.1, vec![1, 2], vec![]),
        ],
        3,
        2,
    );
    // Independent products of the surviving original assignments:
    // C=.0624, B=.1404, E absent=.9, E present=.1.
    for (d1, e, expected_mass, prediction) in [
        (0, 0, 0.05616, mask(&[])),
        (0, 1, 0.01404, mask(&[0, 1])),
        (1, 0, 0.12636, mask(&[0, 1])),
        (1, 1, 0.00624, mask(&[])),
    ] {
        let mut subject = stream(&dem);
        subject.feed_prefix(&[1, d1]).unwrap();
        assert_eq!(subject.advance().unwrap().columns_processed, 3);
        let retired = subject
            .retire_completed_linear_class(&[mask(&[0, 1])])
            .unwrap();
        assert_eq!(retired.selected_class, mask(&[]));
        assert_eq!(retired.states_after, 2);
        assert!(!retired.progress.committed_mask.get(0));
        assert!(!retired.progress.committed_mask.get(1));
        subject.feed_prefix(&[e]).unwrap();
        let result = subject.flush().unwrap();
        assert_eq!(result.predicted, prediction);
        assert_eq!(result.logical_masses.len(), 1);
        assert_eq!(result.logical_masses[0].logical, prediction);
        close(result.logical_masses[0].log_mass.exp(), expected_mass);
        close(result.log_evidence.exp(), expected_mass);
        assert_eq!(result.dropped_states, 0);
    }
}

#[test]
fn wide_input_and_output_coordinates_accept_zero_dependent_and_forced_rows() {
    let dem = model(
        vec![
            (1.0, vec![], vec![64]),
            (0.0, vec![], vec![66]),
            (0.8, vec![], vec![1273]),
        ],
        0,
        1274,
    );
    let mut subject = stream(&dem);
    let mut rows = vec![ObsMask::new(); 130];
    rows[0] = mask(&[1273]);
    rows[64] = mask(&[64]);
    rows[65] = mask(&[1272]); // Never toggled, therefore already closed.
    rows[66] = mask(&[66]); // p=0 is not an unprocessed random column.
    rows[129] = mask(&[1273]); // Dependent, deliberately beyond one output word.
    subject.advance().unwrap();
    let preview = subject.preview_completed_linear_class(&rows).unwrap();
    assert_eq!(preview.selected_class, mask(&[0, 64, 129]));
    assert_eq!(preview.class_log_masses[1].logical, mask(&[64]));
    close(preview.selected_log_mass.exp(), 0.8);
    let retired = subject.retire_completed_linear_class(&rows).unwrap();
    assert_eq!(retired.selected_class, mask(&[0, 64, 129]));
    assert_eq!((retired.states_before, retired.states_after), (2, 1));
    let result = subject.flush().unwrap();
    assert_eq!(result.predicted, mask(&[64, 1273]));
    assert_eq!(result.processed_columns, 1);
    close(result.log_evidence.exp(), 0.8);
    subject.reset();
    let forced = subject
        .retire_completed_linear_class(&[mask(&[64]), mask(&[66]), mask(&[1272])])
        .unwrap();
    assert_eq!(forced.selected_class, mask(&[0]));
    assert_eq!((forced.states_before, forced.states_after), (1, 1));
    assert_eq!(forced.progress.columns_processed, 0);
}

#[test]
fn tied_classes_use_unsigned_compact_row_order_not_full_key_or_row_source_order() {
    let dem = model(
        vec![(0.5, vec![], vec![0]), (0.5, vec![], vec![1273])],
        0,
        1274,
    );
    let mut subject = stream(&dem);
    subject.advance().unwrap();
    let mut rows = vec![ObsMask::new(); 66];
    rows[0] = mask(&[1273]);
    rows[65] = mask(&[0]);
    let preview = subject.preview_completed_linear_class(&rows).unwrap();
    let expected = [mask(&[]), mask(&[0]), mask(&[65]), mask(&[0, 65])];
    assert_eq!(
        preview
            .class_log_masses
            .iter()
            .map(|c| c.logical.clone())
            .collect::<Vec<_>>(),
        expected
    );
    for class in &preview.class_log_masses {
        assert_eq!(
            class.log_mass.to_bits(),
            preview.class_log_masses[0].log_mass.to_bits()
        );
        close(class.log_mass.exp(), 0.25);
    }
    assert_eq!(preview.runner_up_gap.unwrap().to_bits(), 0.0_f64.to_bits());
    let retired = subject.retire_completed_linear_class(&rows).unwrap();
    assert_eq!(retired.selected_class, mask(&[]));
    assert_eq!(subject.flush().unwrap().predicted, mask(&[]));
}

#[test]
fn preview_and_invalid_rows_are_read_only_including_unconsumed_progress() {
    let dem = model(vec![(0.2, vec![], vec![0]), (0.3, vec![], vec![1])], 0, 3);
    let mut subject = stream(&dem);
    let mut control = stream(&dem);
    subject.advance_prediction().unwrap();
    control.advance_prediction().unwrap();
    let before = format!("{subject:?}");
    for _ in 0..3 {
        let preview = subject
            .preview_completed_linear_class(&[mask(&[0, 1])])
            .unwrap();
        assert_eq!(preview.selected_state_count, 2);
        close(preview.selected_log_mass.exp(), 0.62);
        assert_eq!(format!("{subject:?}"), before);
    }
    for rows in [vec![], vec![mask(&[3])], vec![mask(&[]), mask(&[65])]] {
        invalid(&subject.preview_completed_linear_class(&rows).unwrap_err());
        invalid(&subject.retire_completed_linear_class(&rows).unwrap_err());
        assert_eq!(format!("{subject:?}"), before);
    }
    let expected = control.advance().unwrap();
    assert!(expected.newly_committed.contains(&(2, false)));
    assert_eq!(subject.advance().unwrap(), expected);
    result_bits(&subject.flush().unwrap(), &control.flush().unwrap());
}

#[test]
fn unprocessed_sources_and_both_bounded_protocol_states_reject_atomically() {
    let dem = model(vec![(0.2, vec![0], vec![0]), (0.3, vec![1], vec![1])], 2, 2);
    let mut subject = stream(&dem);
    let mut control = stream(&dem);
    for decoder in [&mut subject, &mut control] {
        decoder.feed_prefix(&[1]).unwrap();
        decoder.advance().unwrap();
    }
    let before = format!("{subject:?}");
    for rows in [vec![mask(&[1])], vec![mask(&[]), mask(&[0, 1])]] {
        invalid(&subject.preview_completed_linear_class(&rows).unwrap_err());
        invalid(&subject.retire_completed_linear_class(&rows).unwrap_err());
        assert_eq!(format!("{subject:?}"), before);
    }
    subject.feed_prefix(&[0]).unwrap();
    control.feed_prefix(&[0]).unwrap();
    result_bits(&subject.flush().unwrap(), &control.flush().unwrap());

    let free = model(vec![(0.2, vec![], vec![0])], 0, 1);
    let mut bounded = stream(&free);
    assert_eq!(bounded.begin_bounded_advance().unwrap(), 1);
    for drained in [false, true] {
        if drained {
            assert!(bounded.step_bounded_advance(1).unwrap().1);
        }
        let before = format!("{bounded:?}");
        invalid(
            &bounded
                .preview_completed_linear_class(&[mask(&[])])
                .unwrap_err(),
        );
        invalid(
            &bounded
                .retire_completed_linear_class(&[mask(&[])])
                .unwrap_err(),
        );
        assert_eq!(format!("{bounded:?}"), before);
    }
    bounded.finish_bounded_advance().unwrap();
    assert_eq!(
        bounded
            .retire_completed_linear_class(&[mask(&[])])
            .unwrap()
            .states_after,
        2
    );
}

#[test]
fn zero_projection_keeps_all_states_and_mass_bits_without_merging_full_keys() {
    let dem = model(vec![(0.2, vec![], vec![0]), (0.3, vec![], vec![1])], 0, 2);
    let mut subject = stream(&dem);
    let mut control = stream(&dem);
    subject.advance().unwrap();
    control.advance().unwrap();
    let rows = [mask(&[]), ObsMask::from_words(&[0, 0, 0])];
    let retired = subject.retire_completed_linear_class(&rows).unwrap();
    assert_eq!(retired.selected_class, mask(&[]));
    assert_eq!(
        (
            retired.states_before,
            retired.states_after,
            retired.removed_states
        ),
        (4, 4, 0)
    );
    assert_eq!(retired.class_log_masses.len(), 1);
    assert_eq!(
        retired.log_evidence_before.to_bits(),
        retired.log_evidence_after.to_bits()
    );
    result_bits(&subject.flush().unwrap(), &control.flush().unwrap());
}

#[test]
fn identity_projection_matches_old_retirement_and_bounded_paths_for_all_syndromes() {
    let dem = model(
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
    let rows = [mask(&[0]), mask(&[1]), mask(&[2])];
    for budget in [1, 2, 32] {
        for bits in 0..8_u8 {
            let mut linear = stream(&dem);
            let mut old = stream(&dem);
            let syndrome = [bits & 1, bits >> 1 & 1, bits >> 2 & 1];
            linear.feed_prefix(&syndrome).unwrap();
            old.feed_prefix(&syndrome).unwrap();
            linear.begin_bounded_advance().unwrap();
            while linear.bounded_columns_remaining().unwrap() != 0 {
                linear.step_bounded_advance(budget).unwrap();
            }
            assert_eq!(
                linear.finish_bounded_advance().unwrap(),
                old.advance().unwrap()
            );
            let a = linear.retire_completed_linear_class(&rows).unwrap();
            let b = old.retire_completed_logical_class(&[0, 1, 2]).unwrap();
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
            assert_eq!(
                (a.states_before, a.states_after, a.removed_states),
                (b.states_before, b.states_after, b.removed_states)
            );
            assert_eq!(a.progress, b.progress);
            result_bits(&linear.flush().unwrap(), &old.flush().unwrap());
        }
    }
}

#[test]
fn irreversible_projection_can_regret_and_failure_precedes_bad_rows_until_reset() {
    let dem = model(
        vec![(0.1, vec![0], vec![0]), (0.01, vec![0, 1], vec![])],
        2,
        1,
    );
    let mut subject = stream(&dem);
    subject.feed_prefix(&[1]).unwrap();
    assert_eq!(subject.advance().unwrap().columns_processed, 1);
    assert_eq!(
        subject
            .retire_completed_linear_class(&[mask(&[0])])
            .unwrap()
            .selected_class,
        mask(&[])
    );
    subject.feed_prefix(&[0]).unwrap();
    let failure = subject.advance().unwrap_err();
    assert!(matches!(failure, DecoderError::DecodingFailed(_)));
    for rows in [vec![], vec![mask(&[1])], vec![mask(&[])]] {
        let before = format!("{subject:?}");
        assert_eq!(
            subject
                .preview_completed_linear_class(&rows)
                .unwrap_err()
                .to_string(),
            failure.to_string()
        );
        assert_eq!(
            subject
                .retire_completed_linear_class(&rows)
                .unwrap_err()
                .to_string(),
            failure.to_string()
        );
        assert_eq!(format!("{subject:?}"), before);
    }
    subject.reset();
    subject.feed_dense(&[1, 0]).unwrap();
    let result = subject.flush().unwrap();
    assert_eq!(result.predicted, mask(&[0]));
    let retired = subject
        .retire_completed_linear_class(&[mask(&[0])])
        .unwrap();
    assert_eq!(retired.selected_class, mask(&[0]));
}
