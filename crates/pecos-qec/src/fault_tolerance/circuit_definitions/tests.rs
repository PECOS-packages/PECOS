use super::*;
use pecos_quantum::Gate;

fn emission() -> [Option<MeasId>; 2] {
    [Some(MeasId::from_raw(17)), Some(MeasId::from_raw(9))]
}

fn stamped() -> TickCircuit {
    let mut circuit = TickCircuit::new();
    let mut gate = Gate::mz(&[0, 1]);
    gate.meas_ids
        .extend([MeasId::from_raw(17), MeasId::from_raw(9)]);
    circuit.tick().try_add_gate(gate).unwrap();
    // Match the Python stamped API: reserve through the highest supplied id.
    circuit
        .try_advance_meas_counter(18 - circuit.num_measurements())
        .unwrap();
    circuit
}

fn detector(ids: &[usize]) -> AnnotationDefinition {
    AnnotationDefinition {
        kind: DefinitionKind::Detector { coords: vec![] },
        measurement_ids: ids.iter().copied().map(MeasId::from_raw).collect(),
        label: None,
        pauli: None,
    }
}

fn read(json: &str) -> Result<CircuitDefinitions, DefinitionError> {
    resolve_definitions(Some(json), None, &[], &emission())
}

#[test]
fn out_of_order_stamps() {
    let mut circuit = stamped();
    circuit.set_meta(
        "detectors",
        Attribute::String(
            r#"[{"id":2,"records":[0]},{"id":0,"records":[-2]},{"id":1,"meas_ids":[9]}]"#.into(),
        ),
    );
    let definitions = definitions_from_tick_circuit(&circuit).unwrap();
    // q0/id17 emits first (0), q1/id9 second (1); -2 = 2 - 2 = 0.
    assert_eq!(
        definitions
            .detectors
            .iter()
            .map(|d| (d.id, d.measurements.clone()))
            .collect::<Vec<_>>(),
        vec![(0, vec![0]), (1, vec![1]), (2, vec![0])]
    );
    assert_eq!(definitions.num_measurements, 2);
    let later = circuit.tick().mz(&[2]);
    assert_eq!(later[0].meas_id.index(), 18);
}

#[test]
fn both_forms_agree() {
    for record in [-2, 0] {
        let result = read(&format!(
            r#"[{{"id":0,"records":[{record}],"meas_ids":[17]}}]"#
        ))
        .unwrap();
        // Both -2 and absolute 0 name the first emitted measurement, id17.
        assert_eq!(result.detectors[0].measurements, vec![0]);
    }
    let result = read(r#"[{"id":0,"records":[-1,-2,-1],"meas_ids":[9,9,17]}]"#).unwrap();
    // Metadata record order is retained despite a differently ordered id multiset.
    assert_eq!(result.detectors[0].measurements, vec![1, 0, 1]);
}

#[test]
fn both_forms_disagree() {
    assert!(matches!(
        read(r#"[{"id":0,"records":[-2],"meas_ids":[9]}]"#),
        Err(DefinitionError::ReferenceMismatch { .. })
    ));
}

#[test]
fn both_forms_multiplicity() {
    assert!(matches!(
        read(r#"[{"id":0,"records":[-1,-1],"meas_ids":[9]}]"#),
        Err(DefinitionError::ReferenceMismatch { .. })
    ));
}

#[test]
fn negative_record_bound() {
    assert!(matches!(
        read(r#"[{"id":0,"records":[-3]}]"#),
        Err(DefinitionError::NegativeRecordOutOfRange { record: -3, .. })
    ));
}

#[test]
fn absolute_record_bound() {
    assert!(matches!(
        read(r#"[{"id":0,"records":[2]}]"#),
        Err(DefinitionError::AbsoluteRecordOutOfRange { record: 2, .. })
    ));
    assert!(matches!(
        resolve_definitions(Some(r#"[{"id":0,"records":[0]}]"#), None, &[], &[]),
        Err(DefinitionError::AbsoluteRecordOutOfRange { .. })
    ));
}

#[test]
fn unknown_measurement_id() {
    assert!(matches!(
        read(r#"[{"id":4,"meas_ids":[3]}]"#),
        Err(DefinitionError::UnknownMeasurementId {
            id: 4,
            measurement_id: 3,
            ..
        })
    ));
    assert!(matches!(
        resolve_definitions(None, None, &[detector(&[3])], &emission()),
        Err(DefinitionError::UnknownMeasurementId { id: 0, .. })
    ));
}

#[test]
fn duplicate_metadata_id() {
    let json = r#"[{"id":3,"records":[0]},{"id":3,"records":[1]}]"#;
    assert!(matches!(
        read(json),
        Err(DefinitionError::DuplicateMetadataId {
            kind: "detector",
            id: 3
        })
    ));
    assert!(matches!(
        resolve_definitions(None, Some(json), &[], &emission()),
        Err(DefinitionError::DuplicateMetadataId {
            kind: "observable",
            id: 3
        })
    ));
}

#[test]
fn duplicate_emission_id() {
    assert!(matches!(
        resolve_definitions(
            None,
            None,
            &[],
            &[Some(MeasId::from_raw(9)), None, Some(MeasId::from_raw(9))]
        ),
        Err(DefinitionError::DuplicateEmissionId {
            measurement_id: 9,
            first: 0,
            second: 2
        })
    ));
}

#[test]
fn malformed_metadata() {
    for json in [
        "{",
        r#"[{"id":0}]"#,
        r#"[{"id":0,"records":[0],"label":false}]"#,
    ] {
        assert!(matches!(
            read(json),
            Err(DefinitionError::Parse {
                kind: "detector",
                ..
            })
        ));
        assert!(matches!(
            resolve_definitions(None, Some(json), &[], &emission()),
            Err(DefinitionError::Parse {
                kind: "observable",
                ..
            })
        ));
    }
}

#[test]
fn non_string_attributes() {
    for attribute in ["detectors", "observables", "num_measurements"] {
        let mut circuit = stamped();
        circuit.set_meta(attribute, Attribute::Int(2));
        assert!(
            matches!(definitions_from_tick_circuit(&circuit), Err(DefinitionError::NonStringAttribute { attribute: name, .. }) if name == attribute)
        );
        let dag = DagCircuit::try_from(&circuit).unwrap();
        assert!(
            matches!(definitions_from_dag_circuit(&dag), Err(DefinitionError::NonStringAttribute { attribute: name, .. }) if name == attribute)
        );
    }
}

#[test]
fn invalid_measurement_count() {
    for value in ["two", "-1", "184467440737095516160"] {
        let mut circuit = stamped();
        circuit.set_meta("num_measurements", Attribute::String(value.into()));
        assert!(matches!(
            definitions_from_tick_circuit(&circuit),
            Err(DefinitionError::InvalidMeasurementCount { .. })
        ));
        assert!(matches!(
            definitions_from_dag_circuit(&DagCircuit::try_from(&circuit).unwrap()),
            Err(DefinitionError::InvalidMeasurementCount { .. })
        ));
    }
}

#[test]
fn measurement_count_mismatch() {
    let mut circuit = stamped();
    circuit.set_meta("num_measurements", Attribute::String("18".into()));
    assert!(matches!(
        definitions_from_tick_circuit(&circuit),
        Err(DefinitionError::MeasurementCountMismatch {
            declared: 18,
            actual: 2
        })
    ));
    assert!(matches!(
        definitions_from_dag_circuit(&DagCircuit::try_from(&circuit).unwrap()),
        Err(DefinitionError::MeasurementCountMismatch { .. })
    ));
    circuit.set_meta("num_measurements", Attribute::String("2".into()));
    assert_eq!(
        definitions_from_tick_circuit(&circuit)
            .unwrap()
            .num_measurements,
        2
    );
}

#[test]
fn empty_metadata() {
    for json in ["[]", "", "  \n ", "[ ]"] {
        let mut observable = detector(&[9]);
        observable.kind = DefinitionKind::Observable;
        let result = resolve_definitions(
            Some(json),
            Some(json),
            &[detector(&[17]), observable],
            &emission(),
        )
        .unwrap();
        // id17 is at position 0; id9 at position 1, regardless of empty attributes.
        assert_eq!(result.detectors[0].measurements, vec![0]);
        assert_eq!(result.observables[0].measurements, vec![1]);
        assert_eq!(result.observables[0].pauli, None);
    }
}

#[test]
fn interleaved_annotations() {
    let mut circuit = TickCircuit::new();
    let measurements = circuit.tick().mz(&[0, 1]);
    circuit.tracked_pauli(PauliString::zs(&[0]));
    circuit.observable(&[measurements[1]]).unwrap();
    circuit
        .detector(&[measurements[0], measurements[0]])
        .unwrap();
    circuit.tracked_pauli(PauliString::zs(&[1]));
    circuit.detector(&[measurements[1]]).unwrap();
    circuit.observable(&[measurements[0]]).unwrap();
    let result = definitions_from_tick_circuit(&circuit).unwrap();
    // Measurements of q0,q1 emit at 0,1; tracked Paulis consume neither ids nor records.
    assert_eq!(
        result
            .detectors
            .iter()
            .map(|d| (d.id, d.measurements.clone()))
            .collect::<Vec<_>>(),
        vec![(0, vec![0, 0]), (1, vec![1])]
    );
    assert_eq!(
        result
            .observables
            .iter()
            .map(|d| (d.id, d.measurements.clone()))
            .collect::<Vec<_>>(),
        vec![(0, vec![1]), (1, vec![0])]
    );
    assert_eq!(result.detectors[0].coords, None);
    assert_eq!(result.observables[0].pauli, Some(PauliString::zs(&[1])));
    assert_eq!(
        definitions_from_dag_circuit(&DagCircuit::try_from(&circuit).unwrap()).unwrap(),
        result
    );
}

#[test]
fn merged_fields() {
    let mut annotation = detector(&[9, 17, 9]);
    annotation.kind = DefinitionKind::Detector {
        coords: vec![4.0, 5.0],
    };
    annotation.label = Some("check".into());
    let json = r#"[{"id":0,"records":[-2,-1,-1],"coords":[1,2,3],"label":"check"}]"#;
    let result = resolve_definitions(Some(json), None, &[annotation.clone()], &emission()).unwrap();
    // Metadata order is q0,q1,q1 -> 0,1,1; annotation order is immaterial.
    assert_eq!(
        result.detectors[0],
        ResolvedDetector {
            id: 0,
            coords: Some(vec![1.0, 2.0, 3.0]),
            label: Some("check".into()),
            measurements: vec![0, 1, 1]
        }
    );
    let json = r#"[{"id":0,"records":[0,1,1]}]"#;
    let result = resolve_definitions(Some(json), None, &[annotation.clone()], &emission()).unwrap();
    assert_eq!(result.detectors[0].coords, Some(vec![4.0, 5.0]));
    assert_eq!(result.detectors[0].label.as_deref(), Some("check"));
    annotation.kind = DefinitionKind::Observable;
    annotation.pauli = Some(PauliString::zs(&[0]));
    let result = resolve_definitions(None, Some(json), &[annotation], &emission()).unwrap();
    assert_eq!(result.observables[0].pauli, Some(PauliString::zs(&[0])));
    assert_eq!(result.observables[0].label.as_deref(), Some("check"));
    assert_eq!(result.observables[0].measurements, vec![0, 1, 1]);
}

#[test]
fn source_measurement_mismatch() {
    for ids in [vec![9], vec![17, 17]] {
        assert!(matches!(
            resolve_definitions(
                Some(r#"[{"id":0,"records":[0]}]"#),
                None,
                &[detector(&ids)],
                &emission()
            ),
            Err(DefinitionError::SourceMismatch { .. })
        ));
    }
}

#[test]
fn source_id_mismatch() {
    assert!(matches!(
        resolve_definitions(
            Some(r#"[{"id":1,"records":[0]}]"#),
            None,
            &[detector(&[17])],
            &emission()
        ),
        Err(DefinitionError::IdSetMismatch { .. })
    ));
}

#[test]
fn conflicting_labels() {
    let mut annotation = detector(&[17]);
    annotation.label = Some("annotation".into());
    let json = r#"[{"id":0,"records":[0],"label":"metadata"}]"#;
    assert!(matches!(
        resolve_definitions(Some(json), None, &[annotation.clone()], &emission()),
        Err(DefinitionError::LabelConflict { .. })
    ));
    annotation.label = None;
    let result = resolve_definitions(Some(json), None, &[annotation], &emission()).unwrap();
    assert_eq!(result.detectors[0].label.as_deref(), Some("metadata"));
}

#[test]
fn metadata_only_observables() {
    let result = resolve_definitions(
        None,
        Some(r#"[{"id":9,"meas_ids":[9],"label":"last"},{"id":3,"records":[0]}]"#),
        &[],
        &emission(),
    )
    .unwrap();
    // id3 names position 0; id9 names position 1. Definitions sort by declared id.
    assert_eq!(
        result.observables,
        vec![
            ResolvedObservable {
                id: 3,
                label: None,
                pauli: None,
                measurements: vec![0]
            },
            ResolvedObservable {
                id: 9,
                label: Some("last".into()),
                pauli: None,
                measurements: vec![1]
            }
        ]
    );
}

#[test]
fn dag_keyed_order() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0]);
    dag.pz(&[1]);
    let q0 = dag.mz(&[0]);
    let q1 = dag.mz(&[1]);
    let dfs: Vec<_> = dag
        .topological_order()
        .into_iter()
        .filter(|&node| {
            dag.gate(node)
                .unwrap()
                .gate_type
                .consumes_measurement_record()
        })
        .collect();
    assert_eq!(dfs, vec![q1[0].node, q0[0].node]);
    dag.set_attr(
        "detectors",
        Attribute::String(r#"[{"id":0,"meas_ids":[0]},{"id":1,"meas_ids":[1]}]"#.into()),
    );
    let mut tick = TickCircuit::new();
    tick.tick().pz(&[0]);
    tick.tick().pz(&[1]);
    tick.tick().mz(&[0]);
    tick.tick().mz(&[1]);
    tick.set_meta(
        "detectors",
        Attribute::String(r#"[{"id":0,"meas_ids":[0]},{"id":1,"meas_ids":[1]}]"#.into()),
    );
    for result in [
        definitions_from_dag_circuit(&dag).unwrap(),
        definitions_from_dag_circuit(&DagCircuit::try_from(&tick).unwrap()).unwrap(),
    ] {
        // Node insertion/key order emits q0 before q1, so their positions are 0,1.
        assert_eq!(result.detectors[0].measurements, vec![0]);
        assert_eq!(result.detectors[1].measurements, vec![1]);
    }
}

#[test]
fn tick_dag_emission_parity() {
    let mut circuit = stamped();
    circuit.tick().mx(&[2, 3]);
    circuit.tick();
    circuit
        .ticks_mut()
        .last_mut()
        .unwrap()
        .add_gate(Gate::mz(&[4]));
    circuit.set_meta(
        "detectors",
        Attribute::String(
            r#"[{"id":0,"records":[0,1,2,3,4],"meas_ids":[]},{"id":1,"meas_ids":[9,17,18,19]}]"#
                .into(),
        ),
    );
    let mut dag = DagCircuit::try_from(&circuit).unwrap();
    let tick_result = definitions_from_tick_circuit(&circuit).unwrap();
    assert_eq!(tick_result, definitions_from_dag_circuit(&dag).unwrap());
    // q0,q1 are positions 0,1, MX q2,q3 are 2,3; id-less q4 is position 4.
    assert_eq!(tick_result.detectors[0].measurements, vec![0, 1, 2, 3, 4]);
    assert_eq!(tick_result.detectors[1].measurements, vec![1, 0, 2, 3]);
    assert_eq!(tick_result.num_measurements, 5);
    // DAG conversion reserves 17,9 then 18,19 and mints 20 for q4.
    let minted = dag
        .iter_gates()
        .find(|(_, g)| g.qubits[0].index() == 4)
        .unwrap()
        .1
        .meas_ids[0];
    assert_eq!(minted.index(), 20);
    let json = r#"[{"id":0,"meas_ids":[20]}]"#;
    circuit.set_meta("detectors", Attribute::String(json.into()));
    dag.set_attr("detectors", Attribute::String(json.into()));
    assert!(matches!(
        definitions_from_tick_circuit(&circuit),
        Err(DefinitionError::UnknownMeasurementId {
            measurement_id: 20,
            ..
        })
    ));
    assert_eq!(
        definitions_from_dag_circuit(&dag).unwrap().detectors[0].measurements,
        vec![4]
    );
}

#[test]
fn tick_insertion_holds_measurement_ids_to_one_per_qubit() {
    for ids in [vec![5], vec![5, 6, 7, 8]] {
        let mut circuit = TickCircuit::new();
        let mut gate = Gate::mz(&[0, 1]);
        gate.meas_ids.extend(ids.into_iter().map(MeasId::from_raw));
        // The tick-circuit walk relies on this: a measurement batch carries
        // either no ids or exactly one per qubit.
        assert!(circuit.tick().try_add_gate(gate).is_err());
    }
}

#[test]
fn dag_insertion_holds_measurement_ids_to_one_per_qubit() {
    let mut circuit = DagCircuit::new();
    let mut gate = Gate::mz(&[0, 1]);
    gate.meas_ids.push(MeasId::from_raw(5));
    // The DAG walk relies on this: a measurement node carries either no ids
    // or exactly one per qubit.
    assert!(circuit.try_add_gate(gate).is_err());
}

#[test]
fn observable_source_validation() {
    let mut annotation = detector(&[17]);
    annotation.kind = DefinitionKind::Observable;
    for (json, expected) in [
        (r#"[{"id":0,"records":[1]}]"#, "measurements"),
        (r#"[{"id":2,"records":[0]}]"#, "ids"),
        (r#"[{"id":0,"records":[0],"label":"other"}]"#, "labels"),
    ] {
        annotation.label = Some("annotation".into());
        let error = resolve_definitions(
            None,
            Some(json),
            std::slice::from_ref(&annotation),
            &emission(),
        )
        .unwrap_err();
        assert!(matches!(
            (expected, error),
            (
                "measurements",
                DefinitionError::SourceMismatch {
                    kind: "observable",
                    ..
                }
            ) | (
                "ids",
                DefinitionError::IdSetMismatch {
                    kind: "observable",
                    ..
                }
            ) | (
                "labels",
                DefinitionError::LabelConflict {
                    kind: "observable",
                    ..
                }
            )
        ));
    }
}

#[test]
fn detector_label_parser() {
    for (label, expected) in [("null", None), (r#""name""#, Some("name"))] {
        let parsed =
            parse_detectors_json(&format!(r#"[{{"id":0,"records":[0],"label":{label}}}]"#))
                .unwrap();
        assert_eq!(parsed[0].label.as_deref(), expected);
    }
}

fn with_ids(mut gate: Gate, ids: &[usize]) -> Gate {
    gate.meas_ids
        .extend(ids.iter().copied().map(MeasId::from_raw));
    gate
}

#[test]
fn only_record_consuming_measurements_take_positions() {
    // MZ, MPZ and MeasureFree each consume a record; MeasureLeaked does not,
    // even when it carries a supplied id. Hand-derived emission: ids 10, 12, 13.
    let mut circuit = TickCircuit::new();
    circuit
        .tick()
        .try_add_gate(with_ids(Gate::mz(&[0]), &[10]))
        .unwrap();
    circuit
        .tick()
        .try_add_gate(with_ids(Gate::measure_leaked(&[1]), &[11]))
        .unwrap();
    circuit
        .tick()
        .try_add_gate(with_ids(Gate::mpz(&[2]), &[12]))
        .unwrap();
    circuit
        .tick()
        .try_add_gate(with_ids(
            Gate::simple(
                pecos_quantum::GateType::MeasureFree,
                vec![pecos_quantum::QubitId::from(3usize)],
            ),
            &[13],
        ))
        .unwrap();
    let expected = vec![
        Some(MeasId::from_raw(10)),
        Some(MeasId::from_raw(12)),
        Some(MeasId::from_raw(13)),
    ];
    assert_eq!(tick_circuit_emission(&circuit), expected);
    let dag = DagCircuit::try_from(&circuit).unwrap();
    assert_eq!(dag_circuit_emission(&dag), expected);
}

#[test]
fn merged_batches_emit_in_storage_order() {
    // Within one tick, MZ q0 then MX q1 then MZ q2: the second MZ merges into
    // the first batch, so storage order is [MZ q0 q2], [MX q1] and the records
    // are emitted q0 (id 0), q2 (id 2), q1 (id 1), not in the order added.
    let mut circuit = TickCircuit::new();
    let mut tick = circuit.tick();
    tick.try_add_gate(with_ids(Gate::mz(&[0]), &[0])).unwrap();
    tick.try_add_gate(with_ids(Gate::mx(&[1]), &[1])).unwrap();
    tick.try_add_gate(with_ids(Gate::mz(&[2]), &[2])).unwrap();
    let expected = vec![
        Some(MeasId::from_raw(0)),
        Some(MeasId::from_raw(2)),
        Some(MeasId::from_raw(1)),
    ];
    assert_eq!(tick_circuit_emission(&circuit), expected);
    let dag = DagCircuit::try_from(&circuit).unwrap();
    assert_eq!(dag_circuit_emission(&dag), expected);
}
