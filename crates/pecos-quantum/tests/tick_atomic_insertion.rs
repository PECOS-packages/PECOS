use pecos_quantum::{Gate, QubitId, Tick, TickGateError};

#[test]
fn broadcast_conflicts_are_atomic() {
    let mut tick = Tick::new();
    tick.add_gate(Gate::z(&[1]));
    let before = format!("{tick:?}");
    let result = tick.try_add_gates(vec![Gate::x(&[0]), Gate::x(&[1])]);
    assert!(matches!(result, Err(TickGateError::QubitConflict(_))));
    assert_eq!(format!("{tick:?}"), before);
    assert_eq!(tick.gate_on_qubit(QubitId::from(0)), None);

    // Typed broadcasts use one batch, which must also be checked in full.
    assert!(tick.try_add_gate(Gate::x(&[0, 1])).is_err());
    assert_eq!(format!("{tick:?}"), before);
}

#[test]
fn conflicts_inside_a_broadcast_are_atomic() {
    let mut tick = Tick::new();
    let result = tick.try_add_gates(vec![Gate::cx(&[(0, 1)]), Gate::cx(&[(2, 1)])]);
    assert!(matches!(result, Err(TickGateError::QubitConflict(_))));
    assert!(tick.is_empty());
    assert!(tick.active_qubits().is_empty());
}

#[test]
fn invalid_later_batch_is_atomic() {
    let mut tick = Tick::new();
    assert!(
        tick.try_add_gates(vec![Gate::x(&[0]), Gate::x(&[1, 1])])
            .is_err()
    );
    assert!(tick.is_empty());
    assert!(tick.active_qubits().is_empty());
}

#[test]
fn lookup_tracks_batch_mutations() {
    let mut tick = Tick::new();
    assert_eq!(
        tick.try_add_gates(vec![Gate::h(&[0]), Gate::x(&[1])])
            .unwrap(),
        vec![0, 1]
    );
    tick.add_gate(Gate::h(&[2])); // Coalesces into the first batch.
    assert_eq!(tick.gate_on_qubit(QubitId::from(2)), Some(0));
    tick.replace_gate_batch(0, Gate::h(&[3])).unwrap();
    assert_eq!(tick.gate_on_qubit(QubitId::from(0)), None);
    assert_eq!(tick.gate_on_qubit(QubitId::from(2)), None);
    assert_eq!(tick.gate_on_qubit(QubitId::from(3)), Some(0));
    tick.remove_gate(0);
    assert_eq!(tick.gate_on_qubit(QubitId::from(1)), Some(0));
    assert_eq!(tick.gate_on_qubit(QubitId::from(3)), None);
    tick.discard(&[QubitId::from(1)]);
    assert_eq!(tick.gate_on_qubit(QubitId::from(1)), None);
    tick.try_add_gates(vec![Gate::z(&[1])]).unwrap();
    assert_eq!(tick.gate_on_qubit(QubitId::from(1)), Some(0));
}

#[test]
fn lookup_tracks_metadata_splitting_and_coalescing() {
    use pecos_quantum::{Attribute, TickCircuit};
    let mut circuit = TickCircuit::new();
    circuit
        .tick()
        .h(&[0])
        .h(&[1])
        .meta("label", Attribute::Int(1));
    let tick = circuit.get_tick(0).unwrap();
    assert_eq!(tick.gate_on_qubit(QubitId::from(0)), Some(0));
    assert_eq!(tick.gate_on_qubit(QubitId::from(1)), Some(1));
    circuit.tick_at(0).h(&[2]).meta("label", Attribute::Int(1));
    let tick = circuit.get_tick(0).unwrap();
    assert_eq!(tick.gate_on_qubit(QubitId::from(0)), Some(0));
    assert_eq!(tick.gate_on_qubit(QubitId::from(1)), Some(1));
    assert_eq!(tick.gate_on_qubit(QubitId::from(2)), Some(1));
}

#[test]
fn lookup_tracks_end_middle_and_front_removal() {
    let mut tick = Tick::new();
    tick.try_add_gates((0..5).map(|q| Gate::h(&[q])).collect())
        .unwrap();
    tick.remove_gate(4);
    assert_eq!(tick.gate_on_qubit(QubitId::from(4)), None);
    for q in 0..4 {
        assert_eq!(tick.gate_on_qubit(QubitId::from(q)), Some(q));
    }
    tick.remove_gate(1);
    assert_eq!(tick.gate_on_qubit(QubitId::from(1)), None);
    assert_eq!(tick.gate_on_qubit(QubitId::from(0)), Some(0));
    assert_eq!(tick.gate_on_qubit(QubitId::from(2)), Some(1));
    assert_eq!(tick.gate_on_qubit(QubitId::from(3)), Some(2));
    tick.remove_gate(0);
    assert_eq!(tick.gate_on_qubit(QubitId::from(0)), None);
    assert_eq!(tick.gate_on_qubit(QubitId::from(2)), Some(0));
    assert_eq!(tick.gate_on_qubit(QubitId::from(3)), Some(1));
}

#[test]
fn bulk_discard_keeps_survivor_metadata_and_indices() {
    use pecos_quantum::Attribute;
    let mut tick = Tick::new();
    let indices = tick
        .try_add_gates((0..6).map(|q| Gate::h(&[q])).collect())
        .unwrap();
    for index in indices {
        tick.set_gate_attr(index, "label", Attribute::String(index.to_string()));
    }
    assert_eq!(
        tick.discard(&[QubitId::from(0), QubitId::from(2), QubitId::from(5)]),
        3
    );
    for (index, q) in [1, 3, 4].into_iter().enumerate() {
        assert_eq!(tick.gate_on_qubit(QubitId::from(q)), Some(index));
        assert_eq!(
            tick.get_gate_attr(index, "label"),
            Some(&Attribute::String(q.to_string()))
        );
    }
    for q in [0, 2, 5] {
        assert_eq!(tick.gate_on_qubit(QubitId::from(q)), None);
    }
}

#[test]
fn rejected_custom_gate_does_not_register_signature() {
    use pecos_quantum::TickCircuit;
    let mut circuit = TickCircuit::new();
    circuit.tick().h(&[1]);
    assert!(circuit.tick_at(0).custom_gate("foo", &[0, 1], &[]).is_err());
    assert!(!circuit.gate_signatures().contains_key("foo"));
    circuit.tick_at(0).custom_gate("foo", &[0], &[]).unwrap();
    assert_eq!(circuit.gate_signatures()["foo"].quantum_arity, 1);
}

#[test]
fn bulk_removal_index_matches_stored_batches() {
    let mut tick = Tick::new();
    tick.try_add_gates((0..12).map(|q| Gate::h(&[q])).collect())
        .unwrap();
    for (indices, removed) in [
        (&[0, 4, 11, 4, 99][..], 3),
        (&[1, 3][..], 2),
        (&[0, 1, 2, 3, 4, 5, 6][..], 7),
    ] {
        assert_eq!(tick.remove_gates(indices), removed);
        for q in 0..12 {
            let qubit = QubitId::from(q);
            let expected = tick
                .gate_batches()
                .iter()
                .position(|gate| gate.qubits.contains(&qubit));
            assert_eq!(tick.gate_on_qubit(qubit), expected);
        }
    }
    assert!(tick.is_empty());
}

#[test]
fn discard_both_qubits_counts_one_batch() {
    let mut tick = Tick::new();
    tick.add_gate(Gate::cx(&[(0, 1)]));
    assert_eq!(tick.discard(&[QubitId::from(0), QubitId::from(1)]), 1);
    assert!(tick.is_empty());
}
