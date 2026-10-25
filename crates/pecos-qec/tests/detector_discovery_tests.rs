// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

use pecos_core::Gate;
use pecos_qec::fault_tolerance::detector_discovery::{DetectorDiscoveryError, discover_detectors};
use pecos_qec::fault_tolerance::fault_sampler::{
    MeasurementHistoryError, symbolic_measurement_history,
};
use pecos_quantum::{Attribute, TickCircuit};
use pecos_simulators::{CircuitExecutor, SparseStab};

type TestResult = Result<(), Box<dyn std::error::Error>>;

#[test]
fn repeated_z_check_includes_final_data_readout() -> TestResult {
    let mut tc = TickCircuit::new();
    tc.tick().pz(&[0, 1]);
    for _ in 0..4 {
        tc.tick().pz(&[2]);
        tc.tick().cx(&[(0, 2)]);
        tc.tick().cx(&[(1, 2)]);
        tc.tick().mz(&[2]);
    }
    tc.tick().mz(&[0, 1]);
    let discovered = discover_detectors(&tc, &[])?;
    assert_eq!(
        discovered.detectors,
        vec![
            (vec![0], false),
            (vec![0, 1], false),
            (vec![1, 2], false),
            (vec![2, 3], false),
            (vec![4], false), // Undeclared fixed logical.
            (vec![3, 4, 5], false),
        ]
    );
    let expected = discovered
        .detectors
        .into_iter()
        .filter(|(r, _)| r != &[4])
        .collect::<Vec<_>>();
    // Records 4, 5 and 3^5 are equivalent logical representatives. Each replaces
    // the fixed-logical detector, never the boundary detector [3, 4, 5].
    for observable in [vec![4], vec![5], vec![3, 5]] {
        assert_eq!(
            discover_detectors(&tc, std::slice::from_ref(&observable))?.detectors,
            expected,
            "declared {observable:?}"
        );
    }
    Ok(())
}

fn c4_swapout() -> TickCircuit {
    let mut tc = TickCircuit::new();
    let mut data = [0, 1, 2, 3];
    // Match the prototype's four successive preparation operations as ticks.
    for q in data {
        tc.tick().pz(&[q]);
    }
    for round in 0..4 {
        // Allocate fresh physical ancillas, then relabel the two outgoing data
        // slots as measured ancillas. This is the prototype's SWAP semantics.
        let ap = 4 + 2 * round;
        let a0 = ap + 1;
        let slots = if round % 2 == 0 {
            [0, 1, 2, 3]
        } else {
            [2, 3, 0, 1]
        };
        let [d0, d1, d2, d3] = slots.map(|slot| data[slot]);
        let prep_tick = tc.num_ticks();
        tc.tick().px(&[ap]);
        tc.tick_at(prep_tick).pz(&[a0]);
        for pair in [
            (d0, a0),
            (ap, d1),
            (d1, a0),
            (ap, d0),
            (d2, a0),
            (ap, d3),
            (a0, d3),
            (d2, ap),
        ] {
            tc.tick().cx(&[pair]);
        }
        data[slots[2]] = ap;
        data[slots[3]] = a0;
        let readout_tick = tc.num_ticks();
        tc.tick().mx(&[d2]);
        tc.tick_at(readout_tick).mz(&[d3]);
    }
    tc.tick().mz(&data);
    tc
}

#[test]
fn c4_swapout_checks_and_undeclared_logicals() -> TestResult {
    let tc = c4_swapout();
    let mut checks: Vec<_> = (1..8).map(|j| (vec![j], false)).collect();
    checks.push((vec![8, 9, 10, 11], false));
    let mut with_logicals = checks.clone();
    with_logicals.insert(7, (vec![1, 8, 9], false));
    with_logicals.insert(8, (vec![9, 10], false));
    assert_eq!(discover_detectors(&tc, &[])?.detectors, with_logicals);
    let declared = discover_detectors(&tc, &[vec![8, 9], vec![9, 10]])?;
    assert_eq!(declared.detectors, checks);
    assert_eq!(
        discover_detectors(&tc, &[vec![1, 8, 9], vec![9, 10]])?.detectors,
        checks
    );
    assert_eq!(
        declared.observables,
        vec![(vec![8, 9], false), (vec![9, 10], false)]
    );
    Ok(())
}

#[test]
fn substituted_deterministic_member_contributes_its_nonzero_constant() -> TestResult {
    let mut tc = TickCircuit::new();
    tc.tick();
    tc.tick().pz(&[0, 1, 2]);
    tc.tick().h(&[0]);
    tc.tick().cx(&[(0, 1)]);
    tc.tick().cx(&[(0, 2)]);
    tc.tick().x(&[1]);
    tc.tick().mz(&[0, 1, 2]);
    let full = symbolic_measurement_history(&tc)?;
    assert!(full.as_slice()[1].is_deterministic && full.as_slice()[1].flip);
    assert!(!full.as_slice()[2].flip);
    let found = discover_detectors(&tc, &[])?;
    assert_eq!(
        found.detectors,
        vec![(vec![0, 1], true), (vec![1, 2], true)]
    );
    // Independent, concrete noiseless replay: using only j's constant would
    // return false for the second relation and fail this comparison.
    for _ in 0..32 {
        let shots = CircuitExecutor::new(&tc).run(&mut SparseStab::new(3))?;
        for (records, reference) in &found.detectors {
            let parity = records.iter().fold(false, |p, &r| p ^ shots[r].outcome);
            assert_eq!(*reference, parity);
        }
    }
    Ok(())
}

#[test]
fn implicit_zero_initialization_is_the_search_lower_bound() -> TestResult {
    let mut tc = TickCircuit::new();
    tc.tick().mz(&[0]);
    for _ in 0..17 {
        tc.tick().h(&[1]);
    }
    tc.tick().mz(&[0]);
    assert_eq!(
        discover_detectors(&tc, &[])?.detectors,
        vec![(vec![0], false), (vec![1], false)]
    );
    assert_eq!(
        discover_detectors(&TickCircuit::new(), &[])?.detectors,
        vec![]
    );
    Ok(())
}

#[test]
fn observable_completion_and_replacement() -> TestResult {
    // Data prepared by measurement, then two Z0Z1 rounds: the final d0 readout
    // is deterministic only together with its random preparation record.
    let mut tc = TickCircuit::new();
    tc.tick().h(&[0, 1]);
    tc.tick().mz(&[0, 1]);
    for _ in 0..2 {
        tc.tick().pz(&[2]);
        tc.tick().cx(&[(0, 2)]);
        tc.tick().cx(&[(1, 2)]);
        tc.tick().mz(&[2]);
    }
    tc.tick().mz(&[0, 1]);
    assert_eq!(
        discover_detectors(&tc, &[])?.detectors,
        vec![
            (vec![0, 1, 2], false),
            (vec![2, 3], false),
            (vec![1, 3, 4], false), // Fixed logical Z0.
            (vec![3, 4, 5], false),
        ]
    );
    let checks = vec![
        (vec![0, 1, 2], false),
        (vec![2, 3], false),
        (vec![3, 4, 5], false),
    ];
    // Completion adds each readout's random preparation record; a declared
    // random record cancels against the readout that depends on it. Every
    // representative replaces the fixed logical.
    for (declared, completed) in [
        (vec![4], vec![0, 4]),
        (vec![0, 4], vec![0, 4]),
        (vec![5], vec![1, 5]),
    ] {
        let found = discover_detectors(&tc, std::slice::from_ref(&declared))?;
        assert_eq!(
            found.observables,
            vec![(completed, false)],
            "declared {declared:?}"
        );
        assert_eq!(found.detectors, checks, "declared {declared:?}");
    }
    // The second declaration repeats the first, so only one detector goes.
    let found = discover_detectors(&tc, &[vec![4], vec![4]])?;
    assert_eq!(found.observables, vec![(vec![0, 4], false); 2]);
    assert_eq!(found.detectors, checks);
    Ok(())
}

fn two_patches() -> TickCircuit {
    // Patch A: one Z0Z1 check, then idle ticks. Patch B (qubits 3-5) is
    // prepared later and checked twice. Records 3..6 read 0, 1, 3, 4.
    let mut tc = TickCircuit::new();
    tc.tick().pz(&[0, 1]);
    tc.tick().pz(&[2]);
    tc.tick().cx(&[(0, 2)]);
    tc.tick().cx(&[(1, 2)]);
    tc.tick().mz(&[2]);
    for _ in 0..6 {
        tc.tick();
    }
    tc.tick().pz(&[3, 4]);
    for _ in 0..2 {
        tc.tick().pz(&[5]);
        tc.tick().cx(&[(3, 5)]);
        tc.tick().cx(&[(4, 5)]);
        tc.tick().mz(&[5]);
    }
    tc.tick().mz(&[0, 1, 3, 4]);
    tc
}

/// Patch A's boundary detector [0, 3, 4] has a longer window than patch B's
/// fixed logical [5]. Declaring B's logical times that detector must still
/// replace [5]: dropping [0, 3, 4] would leave X on qubit 1 before readout
/// flipping the observable and no kept detector.
#[test]
fn replacement_keeps_every_single_fault_on_the_observable_detected() -> TestResult {
    let tc = two_patches();
    assert_eq!(
        discover_detectors(&tc, &[])?.detectors,
        vec![
            (vec![0], false),
            (vec![1], false),
            (vec![1, 2], false),
            (vec![3], false),
            (vec![0, 3, 4], false),
            (vec![5], false),
            (vec![2, 5, 6], false),
        ]
    );
    let expected = vec![
        (vec![0], false),
        (vec![1], false),
        (vec![1, 2], false),
        (vec![3], false),
        (vec![0, 3, 4], false),
        (vec![2, 5, 6], false),
    ];
    for observable in [vec![5], vec![0, 3, 4, 5]] {
        assert_eq!(
            discover_detectors(&tc, std::slice::from_ref(&observable))?.detectors,
            expected,
            "declared {observable:?}"
        );
    }
    Ok(())
}

/// Without a check round every discovered detector is a single readout, and
/// any replacement leaves a single X undetected; the protecting checks Z0Z1 and
/// Z1Z2 are not discovered relations, so this is refused, not returned.
#[test]
fn an_observable_no_replacement_can_protect_is_refused() {
    let mut tc = TickCircuit::new();
    tc.tick().pz(&[0, 1, 2]);
    tc.tick().mz(&[0, 1, 2]);
    let error = discover_detectors(&tc, &[vec![0]]);
    assert!(
        matches!(
            &error,
            Err(DetectorDiscoveryError::UndetectableObservableFault {
                observable: 0,
                detector,
                ..
            }) if detector == &[0]
        ),
        "{error:?}"
    );
    if let Err(error) = error {
        let message = error.to_string();
        assert!(message.contains("observable 0") && message.contains("supply detectors by hand"));
    }
}

#[test]
fn observable_errors_name_bad_values() {
    let mut tc = TickCircuit::new();
    tc.tick().h(&[0]);
    tc.tick().mz(&[0]);
    tc.tick().mz(&[0]);
    let nondeterministic = discover_detectors(&tc, &[vec![1], vec![0]]);
    assert!(matches!(
        nondeterministic,
        Err(DetectorDiscoveryError::NonDeterministicObservable {
            observable: 1,
            record: 0
        })
    ));
    if let Err(error) = nondeterministic {
        let message = error.to_string();
        assert!(message.contains("observable 1") && message.contains("record 0"));
    }

    // The reported record is a declared record that is also a random source:
    // here record 2. Record 1 is declared but deterministic, and record 0 is a
    // source that is not declared.
    let mut tc2 = TickCircuit::new();
    tc2.tick().h(&[0, 1]);
    tc2.tick().mz(&[0]);
    tc2.tick().mz(&[0]);
    tc2.tick().mz(&[1]);
    assert!(matches!(
        discover_detectors(&tc2, &[vec![1, 2]]),
        Err(DetectorDiscoveryError::NonDeterministicObservable {
            observable: 0,
            record: 2
        })
    ));
    assert!(matches!(
        discover_detectors(&tc, &[vec![]]),
        Err(DetectorDiscoveryError::EmptyObservable { observable: 0 })
    ));
    let duplicate = discover_detectors(&tc, &[vec![1], vec![1, 0, 1]]);
    assert!(matches!(
        duplicate,
        Err(DetectorDiscoveryError::DuplicateRecord {
            observable: 1,
            record: 1
        })
    ));
    let error = discover_detectors(&tc, &[vec![usize::MAX]]);
    assert!(matches!(
        error,
        Err(DetectorDiscoveryError::RecordOutOfRange {
            observable: 0,
            record: usize::MAX,
            num_records: 2
        })
    ));
    if let Err(error) = error {
        assert!(error.to_string().contains(&usize::MAX.to_string()));
    }
    // Completion accepts this parity (record 0's source cancels), but nothing
    // protects it, so no detector can be replaced for it.
    assert!(matches!(
        discover_detectors(&tc, &[vec![0, 1]]),
        Err(DetectorDiscoveryError::UndetectableObservableFault { observable: 0, .. })
    ));
}

#[test]
fn shared_replay_measurements_and_error_classification() -> TestResult {
    let mut tc = TickCircuit::new();
    tc.tick().pz(&[0, 1]);
    tc.tick().try_add_gate(Gate::measure_leaked(&[3usize]))?;
    tc.tick().px(&[0]);
    tc.tick().mx(&[0]);
    tc.tick().mx(&[0]);
    tc.tick().mz_free(&[1]);
    tc.tick().mpz(&[1]);
    tc.tick().mz(&[1]);
    assert_eq!(
        discover_detectors(&tc, &[])?.detectors,
        vec![
            (vec![0], false),
            (vec![0, 1], false),
            (vec![2], false),
            (vec![2, 3], false),
            (vec![4], false)
        ]
    );
    tc.tick().try_add_gate(Gate::t(&[17]))?;
    assert!(matches!(
        discover_detectors(&tc, &[]),
        Err(DetectorDiscoveryError::MeasurementHistory(
            MeasurementHistoryError::UnsupportedGate(_)
        ))
    ));
    Ok(())
}

#[test]
fn later_qubits_and_long_gaps_do_not_change_local_relations() -> TestResult {
    let mut tc = TickCircuit::new();
    tc.tick().px(&[0]);
    tc.tick().mz(&[0]);
    for _ in 0..37 {
        tc.tick();
    }
    tc.tick().mz(&[0]);
    let expected = vec![(vec![0, 1], false)];
    assert_eq!(discover_detectors(&tc, &[])?.detectors, expected);
    // Ticks after the target measurement are never replayed for it.
    tc.tick().h(&[31]);
    tc.tick().cx(&[(0, 31)]);
    tc.tick().mz(&[31]);
    assert_eq!(discover_detectors(&tc, &[])?.detectors, expected);

    // A high-index qubit first used inside the search window: the purified
    // reference register must cover every qubit the window replays, or this
    // replay goes out of range.
    let mut tc = TickCircuit::new();
    tc.tick().px(&[0]);
    tc.tick().mz(&[0]);
    tc.tick().h(&[31]);
    tc.tick().mz(&[0]);
    assert_eq!(discover_detectors(&tc, &[])?.detectors, expected);
    Ok(())
}

#[test]
fn unrecorded_measure_leaked_outcomes_are_hidden_random_sources() -> TestResult {
    let mut tc = TickCircuit::new();
    tc.tick().pz(&[0]);
    tc.tick().try_add_gate(Gate::measure_leaked(&[0usize]))?;
    tc.tick().mz(&[0]);
    // Every suffix after tick 0 turns the leaked outcome into an unobserved
    // random source, so the record is deterministic only from the real input.
    assert_eq!(
        discover_detectors(&tc, &[])?.detectors,
        vec![(vec![0], false)]
    );
    tc.tick_at(0).h(&[1]);
    tc.tick_at(1)
        .try_add_gate(Gate::measure_leaked(&[1usize]))?;
    tc.tick().mz(&[1]);
    tc.tick().mz(&[1]);
    // The first record to observe the hidden outcome is random; the repeat is
    // deterministic relative to it.
    assert!(!symbolic_measurement_history(&tc)?.as_slice()[1].is_deterministic);
    assert_eq!(
        discover_detectors(&tc, &[])?.detectors,
        vec![(vec![0], false), (vec![1, 2], false)]
    );
    Ok(())
}

/// A qubit measured before a probe boundary and used as a control inside the
/// window is Bell-purified and unmeasured there, so its reset must not
/// postselect (#1360). Postselection made the tick-2 probe report r2 = r1, and
/// that detector fired in about half of noiseless shots.
#[test]
fn reset_after_a_measured_control_inside_a_probe_window() -> TestResult {
    let mut tc = TickCircuit::new();
    tc.tick().h(&[0]);
    tc.tick().mz(&[0]);
    tc.tick().mz(&[1]);
    tc.tick().cx(&[(0, 1)]);
    tc.tick().pz(&[0]);
    tc.tick().mz(&[1]);
    let found = discover_detectors(&tc, &[])?;
    assert_eq!(
        found.detectors,
        vec![(vec![1], false), (vec![0, 1, 2], false)]
    );
    for _ in 0..32 {
        let shots = CircuitExecutor::new(&tc).run(&mut SparseStab::new(2))?;
        for (records, reference) in &found.detectors {
            let parity = records.iter().fold(false, |p, &r| p ^ shots[r].outcome);
            assert_eq!(*reference, parity);
        }
    }
    Ok(())
}

/// Replacing [4] for the first observable leaves the second with only pinned
/// candidates, although replacing [0] and [4] jointly protects both; the search
/// must find that choice whatever the declaration order.
#[test]
fn replacement_is_chosen_jointly_not_in_declaration_order() -> TestResult {
    let mut tc = TickCircuit::new();
    tc.tick().pz(&[0, 1, 2, 3, 4]);
    tc.tick().mz(&[3]);
    tc.tick().cx(&[(2, 3)]);
    tc.tick().cx(&[(0, 1)]);
    tc.tick().cx(&[(3, 4)]);
    tc.tick().mz(&[4]);
    tc.tick().pz(&[4]);
    tc.tick().cx(&[(1, 2)]);
    tc.tick().mz(&[0, 1, 2, 3, 4]);
    let expected = vec![
        (vec![1], false),
        (vec![2], false),
        (vec![3], false),
        (vec![0, 3, 4, 5], false),
        (vec![6], false),
    ];
    for observables in [vec![vec![4, 5], vec![4, 6]], vec![vec![4, 6], vec![4, 5]]] {
        assert_eq!(
            discover_detectors(&tc, &observables)?.detectors,
            expected,
            "declared {observables:?}"
        );
    }
    Ok(())
}

/// Replacement needs only record effects, so the circuit's own detector and
/// observable metadata, stale or not, must not block discovery.
#[test]
fn replacement_ignores_the_circuits_own_definitions() -> TestResult {
    let mut tc = c4_swapout();
    let observables = [vec![8, 9], vec![9, 10]];
    let expected = discover_detectors(&tc, &observables)?;
    tc.set_meta("num_measurements", Attribute::String("99".into()));
    assert_eq!(discover_detectors(&tc, &observables)?, expected);
    Ok(())
}
