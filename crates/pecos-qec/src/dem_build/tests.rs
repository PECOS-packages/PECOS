// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use super::*;
use pecos_core::{Gate, MeasId};
use pecos_decoder_core::obs_mask::ObsMask;
use serde_json::json;

fn reference_one_prepared() -> PreparedDem {
    let mut circuit = TickCircuit::new();
    circuit.tick().x(&[0]);
    let mut measurement = Gate::mz(&[0, 1]);
    measurement
        .meas_ids
        .extend([MeasId::from_raw(7), MeasId::from_raw(3)]);
    circuit.tick().try_add_gate(measurement).unwrap();
    let raw_rows: Vec<_> = (0..8).map(|row| vec![row & 1, (row >> 1) & 1]).collect();
    DetectorErrorModel::builder()
        .with_circuit(circuit)
        .with_detector_definitions(vec![
            DemDefinition::inferred(0, "check", 0).with_metadata("reference", 1),
        ])
        .with_observable_definitions(vec![
            DemDefinition::inferred(70, "logical", 0).with_metadata("reference", 1),
        ])
        .with_inference(DemInferenceSettings::default().with_validation_rows(2))
        .with_inference_data(DemInferenceData {
            raw_measurement_ids: vec![3, 7],
            detector_rows: raw_rows.iter().map(|row| vec![row[1]]).collect(),
            observable_rows: raw_rows.iter().map(|row| vec![row[1]]).collect(),
            raw_rows,
        })
        .prepare()
        .unwrap()
}

#[test]
fn reference_one_all_evaluators_and_raw_variants() {
    let prepared = reference_one_prepared();
    let built = prepared
        .build_dem(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
        .unwrap();
    for build in [&prepared, built.prepared()] {
        let zero = (vec![0], ObsMask::new());
        let mut mask = ObsMask::new();
        mask.set(70);
        let raw = (vec![1], mask);
        let measurements = BTreeMap::from([(7, 1), (999, 255)]);
        let results: BTreeMap<String, Vec<u8>> = BTreeMap::from([
            ("raw measurements".into(), vec![0, 1]),
            ("unbound".into(), vec![255]),
        ]);
        let columns = results
            .iter()
            .map(|(tag, row)| (tag.clone(), vec![row.clone()]))
            .collect();
        assert_eq!(build.evaluate_runtime_record(&[1, 0]).unwrap(), zero);
        assert_eq!(build.evaluate_measurements(&measurements).unwrap(), zero);
        assert_eq!(build.evaluate_results(&results).unwrap(), zero);
        assert_eq!(build.evaluate_result_columns(&columns).unwrap(), vec![zero]);
        assert_eq!(build.runtime_record_parities(&[1, 0]).unwrap(), raw);
        assert_eq!(build.measurement_parities(&measurements).unwrap(), raw);
        assert_eq!(build.result_parities(&results).unwrap(), raw);
        assert_eq!(build.result_column_parities(&columns).unwrap(), vec![raw]);
        assert_eq!(
            definitions_from_tick_circuit(&build.schema.circuit).unwrap(),
            build.schema.definitions
        );
        assert_eq!(build.references().detectors[&0], 1);
        assert_eq!(build.references().observables[&70], 1);
        // Complementing the measured bit produces one event and one flip.
        assert_eq!(build.evaluate_runtime_record(&[0, 0]).unwrap().0, [1]);
        assert!(build.evaluate_runtime_record(&[0, 0]).unwrap().1.get(70));
    }
}

#[test]
fn evaluator_validation_holes_aliases_and_columns() {
    let original = reference_one_prepared();
    let traces = [
        bindings::ResultTrace {
            name: "a".into(),
            occurrence: Some(0),
            width: 1,
            measurement_ids: Some(vec![7]),
        },
        bindings::ResultTrace {
            name: "a".into(),
            occurrence: Some(1),
            width: 1,
            measurement_ids: None,
        },
        bindings::ResultTrace {
            name: "alias".into(),
            occurrence: Some(0),
            width: 1,
            measurement_ids: Some(vec![7]),
        },
        bindings::ResultTrace {
            name: "absent".into(),
            occurrence: Some(0),
            width: 1,
            measurement_ids: Some(vec![3]),
        },
    ];
    let schema = DemBuildSchema::new(
        original.schema.circuit.clone(),
        original.schema.definitions.clone(),
        original.schema.measurement_ids.clone(),
        &traces,
        None,
    )
    .unwrap();
    let build = PreparedDem {
        schema: Arc::new(schema),
        ..original
    };
    assert_eq!(build.named_result_measurements()["a"], [Some(7), None]);
    assert_eq!(build.audit().measurement_ledger, build.measurement_ledger());
    let good = BTreeMap::from([
        ("a".into(), vec![1, 255]),
        ("alias".into(), vec![1]),
        ("ignored".into(), vec![255]),
    ]);
    assert_eq!(build.evaluate_results(&good).unwrap().0, [0]);
    for raw in [false, true] {
        let runtime = |values: &[u8]| {
            if raw {
                build.runtime_record_parities(values)
            } else {
                build.evaluate_runtime_record(values)
            }
        };
        assert_eq!(
            runtime(&[1]).unwrap_err().to_string(),
            "runtime measurement record has 1 values; expected 2"
        );
        assert_eq!(
            runtime(&[1, 9]).unwrap_err().to_string(),
            "runtime measurement 1 must be bool, 0, or 1; got 9"
        );
        let measurements = |values: &BTreeMap<usize, u8>| {
            if raw {
                build.measurement_parities(values)
            } else {
                build.evaluate_measurements(values)
            }
        };
        assert_eq!(
            measurements(&BTreeMap::new()).unwrap_err().to_string(),
            "measurement values are missing referenced MeasId(s): [7]"
        );
        assert_eq!(
            measurements(&BTreeMap::from([(7, 2)]))
                .unwrap_err()
                .to_string(),
            "measurement MeasId 7 must be bool, 0, or 1; got 2"
        );
        let results = |values: &BTreeMap<String, Vec<u8>>| {
            if raw {
                build.result_parities(values)
            } else {
                build.evaluate_results(values)
            }
        };
        assert_eq!(
            results(&BTreeMap::from([("a".into(), vec![1])]))
                .unwrap_err()
                .to_string(),
            "result \"a\" has 1 values; trace provenance expects 2"
        );
        assert_eq!(
            results(&BTreeMap::from([("a".into(), vec![2, 255])]))
                .unwrap_err()
                .to_string(),
            "result \"a\" must be bool, 0, or 1; got 2"
        );
        let mut contradictory = good.clone();
        contradictory.insert("alias".into(), vec![0]);
        assert_eq!(
            results(&contradictory).unwrap_err().to_string(),
            "result outputs disagree about measurement MeasId 7"
        );
        let columns = |values: &BTreeMap<String, Vec<Vec<u8>>>| {
            if raw {
                build.result_column_parities(values)
            } else {
                build.evaluate_result_columns(values)
            }
        };
        assert_eq!(
            columns(&BTreeMap::from([("ignored".into(), vec![])]))
                .unwrap_err()
                .to_string(),
            "result columns do not contain any bound measurement tags"
        );
        assert_eq!(
            columns(&BTreeMap::from([
                ("a".into(), vec![]),
                ("ignored".into(), vec![vec![255]])
            ]))
            .unwrap(),
            Vec::<(Vec<u8>, ObsMask)>::new()
        );
        assert_eq!(
            columns(&BTreeMap::from([
                ("a".into(), vec![vec![1, 255]]),
                ("alias".into(), vec![])
            ]))
            .unwrap_err()
            .to_string(),
            "result columns have inconsistent shot counts: {\"alias\": 0}"
        );
    }
}

#[test]
fn references_reject_random_roles_but_readers_do_not() {
    for role in ["detectors", "observables"] {
        let mut circuit = TickCircuit::new();
        circuit.tick().h(&[0]);
        circuit.tick().mz(&[0]);
        circuit.set_meta(
            role,
            Attribute::String(r#"[{"id":0,"meas_ids":[0],"reference":1}]"#.into()),
        );
        let definitions = definitions_from_tick_circuit(&circuit).unwrap();
        assert_eq!(definitions.num_measurements, 1);
        assert!(
            DemBuilder::try_from_tick_circuit_with_noise_config(
                &circuit,
                NoiseConfig::new(0.001, 0.01, 0.001, 0.001)
            )
            .is_ok()
        );
        let error = DetectorErrorModel::builder()
            .with_circuit(circuit)
            .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
            .build()
            .err()
            .unwrap();
        let singular = role.trim_end_matches('s');
        assert_eq!(
            error.to_string(),
            format!(
                "{singular} 0 has a random noiseless parity; detectors and observables must be deterministic in the noiseless circuit"
            )
        );
    }
}

#[test]
fn reference_one_does_not_license_affine_inference() {
    let mut data = DemInferenceData {
        raw_measurement_ids: vec![0],
        raw_rows: vec![vec![0], vec![1], vec![0], vec![1]],
        detector_rows: vec![vec![1], vec![0], vec![1], vec![0]],
        observable_rows: vec![vec![]; 4],
    };
    assert!(matches!(
        DemInferenceSettings::default()
            .with_validation_rows(1)
            .infer(&data),
        Err(DemInferenceError::ConstantOne)
    ));
    data.detector_rows = vec![vec![0], vec![1], vec![0], vec![1]];
    assert!(
        DemInferenceSettings::default()
            .with_validation_rows(1)
            .infer(&data)
            .is_ok()
    );
}

#[test]
fn transparent_gates_leave_references_unchanged() {
    use crate::fault_tolerance::propagator::is_supported_noop_or_metadata_gate;
    use pecos_core::gate_type::GateType;
    let mut circuit = TickCircuit::new();
    circuit.tick().x(&[0]);
    circuit.tick().mz(&[0]);
    circuit.set_meta(
        "detectors",
        Attribute::String(r#"[{"id":0,"meas_ids":[0]}]"#.into()),
    );
    circuit.set_meta(
        "observables",
        Attribute::String(r#"[{"id":70,"meas_ids":[0]}]"#.into()),
    );
    let definitions = definitions_from_tick_circuit(&circuit).unwrap();
    let expected = noiseless_references(&circuit, &definitions).unwrap();
    for gate_type in [
        GateType::I,
        GateType::Idle,
        GateType::QFree,
        GateType::MeasCrosstalkGlobalPayload,
        GateType::MeasCrosstalkLocalPayload,
        GateType::TrackedPauliMeta,
    ] {
        assert!(is_supported_noop_or_metadata_gate(gate_type));
        let mut with_gate = TickCircuit::new();
        with_gate.tick().x(&[0]);
        with_gate
            .tick()
            .try_add_gate(Gate::new(
                gate_type,
                vec![],
                vec![0.0; gate_type.classical_arity()],
                (0..gate_type.quantum_arity())
                    .map(pecos_core::QubitId)
                    .collect::<Vec<_>>(),
            ))
            .unwrap();
        with_gate.tick().mz(&[0]);
        assert_eq!(
            noiseless_references(&with_gate, &definitions).unwrap(),
            expected
        );
    }
}

#[test]
fn audit_tracks_order_bindings_references_and_fingerprint() {
    let build = reference_one_prepared();
    let audit = build.audit();
    assert_eq!(audit.runtime_measurement_order, [7, 3]);
    assert!(!audit.runtime_order_is_canonical);
    assert_eq!(audit.runtime_order_mismatch_count, 2);
    assert_eq!(audit.named_result_binding, "caller_bound_measurement_ids");
    assert_eq!(audit.soundness_scope, "empirical_static_schedule");
    assert!(
        audit
            .measurement_ledger
            .iter()
            .all(|entry| entry.canonical_record_index.is_none())
    );
    assert_eq!(
        audit.measurement_ledger[0].result_refs[0].element(),
        Some(1)
    );
    let value = audit.to_json();
    assert_eq!(value.as_object().unwrap().len(), 14);
    assert_eq!(value["inference"].as_object().unwrap().len(), 9);
    assert_eq!(
        value["definition_origins"][1],
        json!({"role": "observable", "id":70, "source":"inferred", "tag":"logical", "element":0})
    );
    assert_eq!(value["inference"]["provenance_shots"], json!(null));
    let read = |key| {
        let Some(Attribute::String(value)) = build.schema.circuit.get_meta(key) else {
            panic!("metadata");
        };
        serde_json::from_str(value).unwrap()
    };
    let detectors = read("detectors");
    let observables = read("observables");
    let fingerprint = schema_fingerprint(
        &detectors,
        &observables,
        &build.schema.measurement_ids,
        &build.schema.named_result_measurements,
    )
    .unwrap();
    assert_eq!(build.schema.schema_fingerprint, fingerprint);
    assert_eq!(
        build.schema.circuit.get_meta("dem_schema_fingerprint"),
        Some(&Attribute::String(fingerprint.clone()))
    );
    for (order, bindings) in [
        (vec![3, 7], build.schema.named_result_measurements.clone()),
        (
            vec![7, 3],
            BTreeMap::from([("renamed".into(), vec![Some(3), Some(7)])]),
        ),
    ] {
        assert_ne!(
            schema_fingerprint(&detectors, &observables, &order, &bindings).unwrap(),
            fingerprint
        );
    }
}

#[test]
fn residual_summary_strict_threshold_sorted_kinds_and_maxima() {
    use crate::fault_tolerance::dem_builder::{
        FaultMechanism, NoiseChannelKind, NoiseChannelResidual,
    };
    let make = |channel_kind, magnitude, channel_weight| NoiseChannelResidual {
        location_index: 0,
        channel_kind,
        effect: FaultMechanism::default(),
        magnitude,
        channel_weight,
    };
    let residuals = [
        make(NoiseChannelKind::TwoQubitGate, 0.2, 0.5),
        make(NoiseChannelKind::Idle, 0.1, 0.2),
        make(NoiseChannelKind::Idle, 0.15, 0.5),
        make(NoiseChannelKind::SingleQubitGate, 0.125, 0.5),
        make(NoiseChannelKind::SingleQubitGate, f64::NAN, 0.5),
    ];
    let summary = summarize_noise_residuals(&residuals, 0.25).unwrap();
    assert_eq!(
        summary,
        vec![
            NoiseResidualSummary {
                channel_kind: "idle",
                count: 2,
                max_relative_magnitude: 0.5,
                max_tv_magnitude: 0.15
            },
            NoiseResidualSummary {
                channel_kind: "two-qubit gate",
                count: 1,
                max_relative_magnitude: 0.4,
                max_tv_magnitude: 0.2
            }
        ]
    );
    assert_eq!(
        summarize_noise_residuals(&residuals, 1.0).unwrap(),
        Vec::<NoiseResidualSummary>::new()
    );
    assert_eq!(summarize_noise_residuals(&residuals, 0.0).unwrap().len(), 3);
    for threshold in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(summarize_noise_residuals(&[], threshold).is_err());
    }
}

#[test]
fn reference_metadata_preserves_aliases_custom_fields_and_annotations() {
    let mut circuit = TickCircuit::new();
    circuit.tick().x(&[0]);
    circuit.tick().mz(&[0]);
    let measured = circuit.meas_ref(1, 0, pecos_core::QubitId(0)).unwrap();
    circuit.detector(&[measured]).unwrap();
    circuit.set_meta("detectors", Attribute::String(r#"[{"detector_id":"D0","records":[-1],"reference":null,"custom":{"nested":0.12345678901234567}}]"#.into()));
    let build = DetectorErrorModel::builder()
        .with_circuit(circuit)
        .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
        .build()
        .unwrap();
    let build = build.prepared();
    let Some(Attribute::String(text)) = build.schema.circuit.get_meta("detectors") else {
        panic!("metadata");
    };
    let value: serde_json::Value = serde_json::from_str(text).unwrap();
    assert_eq!(value[0]["detector_id"], "D0");
    assert_eq!(value[0]["records"], json!([-1]));
    assert_eq!(value[0]["reference"], 1);
    assert_eq!(
        value[0]["custom"]["nested"].as_f64(),
        Some(0.123_456_789_012_345_66)
    );
    assert_eq!(
        definitions_from_tick_circuit(&build.schema.circuit).unwrap(),
        build.schema.definitions
    );
    let audit = build.audit();
    assert_eq!(audit.soundness_scope, "static_schedule_only");
    assert_eq!(audit.named_result_binding, "none");
    assert!(audit.runtime_order_is_canonical);
    assert_eq!(audit.runtime_order_mismatch_count, 0);
    assert!(audit.inference.is_none());
}

#[test]
fn references_preserve_sparse_identity_mapping_and_support_multiplicity() {
    let mut circuit = TickCircuit::new();
    circuit.tick().h(&[0]);
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().x(&[1]);
    let mut measured = Gate::mz(&[1, 0]);
    measured
        .meas_ids
        .extend([MeasId::from_raw(17), MeasId::from_raw(9)]);
    circuit.tick().try_add_gate(measured).unwrap();
    let build = DetectorErrorModel::builder()
        .with_circuit(circuit)
        .with_detectors_json(
            r#"[{"id":0,"records":[-1,-2],"meas_ids":[9,17]},{"id":1,"meas_ids":[17,17]}]"#,
        )
        .with_observables_json(r#"[{"id":70,"records":[0,1]}]"#)
        .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
        .build()
        .unwrap();
    let build = build.prepared();
    assert_eq!(
        build.references().detectors,
        BTreeMap::from([(0, 1), (1, 0)])
    );
    assert_eq!(build.references().observables, BTreeMap::from([(70, 1)]));
    for record in [[0, 1], [1, 0]] {
        assert_eq!(
            build.evaluate_runtime_record(&record).unwrap(),
            (vec![0, 0], ObsMask::new())
        );
    }
}

#[test]
fn references_require_matching_history_length() {
    let mut circuit = TickCircuit::new();
    circuit.tick().mz(&[0]);
    let mut definitions = definitions_from_tick_circuit(&circuit).unwrap();
    for count in [0, 2] {
        definitions.num_measurements = count;
        let error = noiseless_references(&circuit, &definitions).unwrap_err();
        assert!(matches!(error, DemBuildError::Configuration(_)));
        assert_eq!(
            error.to_string(),
            format!("symbolic measurement history has 1 records; definitions expect {count}")
        );
    }
}

#[test]
fn unsupported_reference_gate_preserves_typed_propagation_error() {
    use crate::fault_tolerance::propagator::UnsupportedGateLocation;
    use pecos_core::gate_type::GateType;

    let mut circuit = TickCircuit::new();
    circuit.tick().t(&[0]);
    circuit.tick().mz(&[0]);
    let error = DetectorErrorModel::builder()
        .with_circuit(circuit)
        .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
        .build()
        .err()
        .unwrap();
    let DemBuildError::Propagation(DemBuilderError::UnsupportedGate(error)) = error else {
        panic!("expected typed unsupported gate error, got {error}");
    };
    assert_eq!(error.gate_type, GateType::T);
    assert_eq!(error.qubits, [0]);
    assert_eq!(
        error.location,
        UnsupportedGateLocation::Tick {
            tick: 0,
            gate_in_tick: 0
        }
    );
}

#[test]
fn prepare_needs_no_noise_and_ignores_supplied_noise() {
    let builder = || {
        let mut circuit = TickCircuit::new();
        circuit.tick().mz(&[0]);
        DetectorErrorModel::builder()
            .with_circuit(circuit)
            .with_detectors(vec![vec![0]])
    };
    let prepared = builder().prepare().unwrap();
    let ignored = builder()
        .with_noise(NoiseConfig::with_idle(0.0, 0.0, 0.0, 0.0, 0.1))
        .prepare()
        .unwrap();
    assert_eq!(prepared.audit(), ignored.audit());
    assert_eq!(prepared.definitions(), ignored.definitions());
    assert_eq!(
        builder().build().err().unwrap().to_string(),
        "supply a noise configuration with with_noise"
    );
}

#[test]
fn repeated_dem_builds_preserve_prepared_schema_and_circuit() {
    let prepared = reference_one_prepared();
    let before = prepared.circuit().clone();
    let audit = prepared.audit();
    let first = prepared
        .build_dem(NoiseConfig::new(0.0, 0.0, 0.01, 0.0))
        .unwrap();
    let second = prepared
        .build_dem(NoiseConfig::new(0.0, 0.0, 0.2, 0.0))
        .unwrap();
    assert_ne!(first.dem().to_string(), second.dem().to_string());
    for build in [&first, &second] {
        assert!(std::ptr::eq(build.prepared().circuit(), prepared.circuit()));
        assert_eq!(build.prepared().audit(), audit);
        assert_eq!(build.prepared().definitions(), prepared.definitions());
        assert_eq!(build.prepared().references(), prepared.references());
        assert_eq!(
            build.prepared().schema_fingerprint(),
            prepared.schema_fingerprint()
        );
        assert_eq!(
            build.prepared().definition_origins(),
            prepared.definition_origins()
        );
        assert_eq!(build.prepared().inference(), prepared.inference());
        assert_eq!(
            format!("{:?}", build.prepared().circuit()),
            format!("{before:?}")
        );
        assert_eq!(
            build.idle_noise_residuals(),
            build.dem().idle_noise_residuals()
        );
        assert_eq!(
            build.residual_summary(0.0).unwrap(),
            summarize_noise_residuals(build.idle_noise_residuals(), 0.0).unwrap()
        );
    }
    // TickCircuit has no PartialEq; its Debug representation includes all state.
    assert_eq!(format!("{:?}", prepared.circuit()), format!("{before:?}"));
    assert_eq!(prepared.audit(), audit);
}

#[test]
fn failed_idle_dem_build_leaves_preparation_reusable() {
    let prepared = reference_one_prepared();
    let before = prepared.circuit().clone();
    let mut coherent = NoiseConfig::new(0.0, 0.0, 0.0, 0.0);
    coherent.idle_rz = 0.1;
    for noise in [NoiseConfig::with_idle(0.0, 0.0, 0.0, 0.0, 0.1), coherent] {
        assert_eq!(
            prepared.build_dem(noise).err().unwrap().to_string(),
            "idle-noise parameters have no idle gates to attach to"
        );
        assert!(
            prepared
                .build_dem(NoiseConfig::new(0.0, 0.0, 0.01, 0.0))
                .is_ok()
        );
        assert_eq!(format!("{:?}", prepared.circuit()), format!("{before:?}"));
    }
    assert!(
        prepared
            .build_dem(NoiseConfig::with_idle(0.0, 0.0, 0.01, 0.0, 0.0))
            .is_ok()
    );
}

#[test]
fn unified_references_prepare_records_and_cancel_support() {
    let prepare = |refs| {
        let mut circuit = TickCircuit::new();
        circuit.tick().mz(&[0, 1]);
        DetectorErrorModel::builder()
            .with_circuit(circuit)
            .with_detector_definitions(vec![DemDefinition::references(0, refs).unwrap()])
            .with_observable_definitions(vec![
                DemDefinition::references(70, vec![MeasurementRef::record(1).unwrap()]).unwrap(),
            ])
            .prepare()
    };
    let prepared = prepare(vec![
        MeasurementRef::record(2).unwrap(),
        MeasurementRef::record(1).unwrap(),
        MeasurementRef::record(2).unwrap(),
    ])
    .unwrap();
    assert_eq!(prepared.definitions().detectors[0].measurements, [1]);
    assert_eq!(prepared.definitions().observables[0].measurements, [1]);
    assert_eq!(
        prepare(vec![
            MeasurementRef::record(1).unwrap(),
            MeasurementRef::record(1).unwrap()
        ])
        .err()
        .unwrap()
        .to_string(),
        "detector 0 references every measurement with even multiplicity; its parity is identically zero"
    );
    assert_eq!(
        prepare(vec![MeasurementRef::tag("missing").unwrap()])
            .err()
            .unwrap()
            .to_string(),
        "result_ref \"missing\" occurrence 0 is absent from the runtime trace"
    );
}

#[test]
fn unified_result_reference_uses_available_inference_trace() {
    let prepare = |refs| {
        let mut circuit = TickCircuit::new();
        let mut measurement = Gate::mz(&[0]);
        measurement.meas_ids.push(MeasId::from_raw(0));
        circuit.tick().try_add_gate(measurement).unwrap();
        let rows = vec![vec![0], vec![1], vec![0], vec![1]];
        DetectorErrorModel::builder()
            .with_circuit(circuit)
            .with_detector_definitions(vec![DemDefinition::inferred(0, "check", 0)])
            .with_observable_definitions(vec![DemDefinition::references(70, refs).unwrap()])
            .with_inference(DemInferenceSettings::default().with_validation_rows(1))
            .with_inference_data(DemInferenceData {
                raw_measurement_ids: vec![0],
                detector_rows: rows.clone(),
                observable_rows: vec![vec![]; rows.len()],
                raw_rows: rows,
            })
            .prepare()
    };
    let named = MeasurementRef::Result(ResultRef::new("raw measurements", 0, Some(0)).unwrap());
    let prepared = prepare(vec![
        named.clone(),
        MeasurementRef::record(1).unwrap(),
        named.clone(),
    ])
    .unwrap();
    assert_eq!(prepared.definitions().observables[0].measurements, [0]);
    assert_eq!(
        prepared.named_result_measurements()["raw measurements"],
        [Some(0)]
    );
    assert_eq!(
        prepare(vec![named, MeasurementRef::record(1).unwrap()])
            .err()
            .unwrap()
            .to_string(),
        "observable 70 references every measurement with even multiplicity; its parity is identically zero"
    );
}

#[test]
fn declared_measurement_count_is_checked_before_definition_resolution() {
    let circuit = || {
        let mut circuit = TickCircuit::new();
        circuit.tick().mz(&[0]);
        circuit
    };
    let prepared = DetectorErrorModel::builder()
        .with_circuit(circuit())
        .with_detectors_json(r#"[{"id":0,"records":[-1]}]"#)
        .with_num_measurements(1)
        .prepare()
        .unwrap();
    assert_eq!(prepared.measurement_ids(), [0]);
    let error = DetectorErrorModel::builder()
        .with_circuit(circuit())
        .with_detectors_json(r#"[{"id":0,"records":[-2]}]"#)
        .with_num_measurements(2)
        .prepare()
        .err()
        .unwrap();
    assert!(matches!(
        error,
        DemBuildError::Definitions(DefinitionError::MeasurementCountMismatch {
            declared: 2,
            actual: 1
        })
    ));
    for builder in [
        DetectorErrorModel::builder().with_inference(DemInferenceSettings::default()),
        DetectorErrorModel::builder().with_detector_definitions(vec![]),
        DetectorErrorModel::builder().with_observable_definitions(vec![]),
    ] {
        assert!(
            builder
                .with_num_measurements(1)
                .prepare()
                .err()
                .unwrap()
                .to_string()
                .contains("cannot be combined")
        );
    }
}

fn idle_circuit(zero_rz: bool) -> TickCircuit {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0, 1]);
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().idle(3, &[0]);
    if zero_rz {
        circuit.tick().rz(pecos_core::Angle64::ZERO, &[0]);
    }
    circuit.tick().mz(&[0, 1]);
    circuit
}

#[test]
fn native_idle_pass_policy_preserves_targets_and_zero_rz_fault_sites() {
    use pecos_core::gate_type::GateType;

    let builder = |zero_rz| {
        DetectorErrorModel::builder()
            .with_circuit(idle_circuit(zero_rz))
            .with_detectors(vec![vec![0]])
    };
    let existing = builder(true).prepare().unwrap();
    let inserted = builder(true).with_idle_after_2q(1.0).prepare().unwrap();
    let kept = builder(true)
        .with_idle_after_2q(1.0)
        .with_strip_traced_idles(false)
        .prepare()
        .unwrap();
    let stripped = builder(true)
        .with_strip_traced_idles(true)
        .prepare()
        .unwrap();
    let idle_targets = |prepared: &PreparedDem| {
        prepared
            .circuit()
            .iter_gate_batches()
            .filter(|gate| gate.as_gate().gate_type == GateType::Idle)
            .map(|gate| gate.as_gate().qubits.len())
            .sum::<usize>()
    };
    assert_eq!(idle_targets(&existing), 1);
    assert_eq!(idle_targets(&inserted), 2);
    assert_eq!(idle_targets(&kept), 3);
    assert_eq!(idle_targets(&stripped), 0);
    let idle_noise = NoiseConfig::with_idle(0.0, 0.0, 0.0, 0.0, 0.1);
    assert!(existing.build_dem(idle_noise.clone()).is_ok());
    assert_ne!(
        inserted
            .build_dem(idle_noise.clone())
            .unwrap()
            .dem()
            .to_string(),
        kept.build_dem(idle_noise.clone())
            .unwrap()
            .dem()
            .to_string()
    );
    assert!(stripped.build_dem(idle_noise).is_err());
    let p1 = NoiseConfig::new(0.09, 0.0, 0.0, 0.0);
    for prepared in [&existing, &inserted, &kept, &stripped] {
        assert_eq!(
            prepared
                .circuit()
                .iter_gate_batches()
                .filter(|gate| gate.as_gate().gate_type == GateType::RZ)
                .count(),
            1
        );
        assert!(
            prepared
                .build_dem(p1.clone())
                .unwrap()
                .dem()
                .to_string()
                .contains("error(")
        );
    }
    let without_rz = builder(false).with_idle_after_2q(1.0).prepare().unwrap();
    assert!(
        !without_rz
            .build_dem(p1)
            .unwrap()
            .dem()
            .to_string()
            .contains("error(")
    );
}

#[test]
fn preparation_restores_measurement_count_and_inference_method_metadata() {
    let inferred = reference_one_prepared();
    let explicit = DetectorErrorModel::builder()
        .with_circuit(idle_circuit(false))
        .with_detectors(vec![vec![0]])
        .prepare()
        .unwrap();
    for prepared in [&inferred, &explicit] {
        assert_eq!(
            prepared.circuit().get_meta("num_measurements"),
            Some(&Attribute::String("2".into()))
        );
    }
    assert_eq!(
        inferred.circuit().get_meta("guppy_dem_annotation_method"),
        Some(&Attribute::String("coin_toss_affine_inference_v2".into()))
    );
    assert!(
        explicit
            .circuit()
            .get_meta("guppy_dem_annotation_method")
            .is_none()
    );
}

#[test]
fn preparation_does_not_overwrite_an_invalid_declared_circuit_count() {
    let mut circuit = idle_circuit(false);
    circuit.set_meta("num_measurements", Attribute::String("99".into()));
    let error = DetectorErrorModel::builder()
        .with_circuit(circuit)
        .with_detectors(vec![vec![0]])
        .prepare()
        .err()
        .unwrap();
    assert!(matches!(
        error,
        DemBuildError::Definitions(DefinitionError::MeasurementCountMismatch {
            declared: 99,
            actual: 2
        })
    ));
}
