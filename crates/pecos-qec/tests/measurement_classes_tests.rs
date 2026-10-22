// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use pecos_core::MeasId;
use pecos_qec::fault_tolerance::DagFaultInfluenceMap;
use pecos_qec::{DemBuilder, MemoryBasis, ParityCheckMatrix, coloration_memory_circuit};
use pecos_quantum::{Attribute, DagCircuit, Gate, TickCircuit};
use serde_json::{Value, json};

fn class(family: &Value) -> Value {
    json!({"block": "memory", "family": family, "round": -1})
}

fn circuit(classes: &Value, records: &Value, explicit: &Value) -> TickCircuit {
    let mut circuit = TickCircuit::new();
    circuit.tick().mz(&[0, 1, 2]);
    circuit.set_meta(
        "measurement_classes",
        Attribute::String(classes.to_string()),
    );
    circuit.set_meta(
        "detectors",
        Attribute::String(
            json!([
                {"id": 3, "records": records, "fault_sector": explicit}
            ])
            .to_string(),
        ),
    );
    circuit
}

#[test]
fn circuit_classes_derive_concrete_labels_and_survive_conversion() {
    for records in [json!([0, 1]), json!([2, 2, 0]), json!([-1, 2, 0])] {
        let circuit = circuit(
            &json!([class(&json!("X")), class(&json!("X")), class(&json!("Z"))]),
            &records,
            &Value::Null,
        );
        let dag = DagCircuit::try_from(&circuit).unwrap();
        for circuit in [circuit.clone(), TickCircuit::from(&dag)] {
            let dem = DemBuilder::from_tick_circuit(&circuit, 0.0, 0.0, 0.0, 0.0).unwrap();
            assert_eq!(dem.detectors[0].fault_sector.as_deref(), Some("X"));
        }
        let dem = DemBuilder::from_circuit(&dag, 0.0, 0.0, 0.0, 0.0).unwrap();
        assert_eq!(dem.detectors[0].fault_sector.as_deref(), Some("X"));
    }
}

#[test]
fn unknown_derived_sectors_reject_explicit_labels() {
    for (classes, records) in [
        (
            json!([class(&json!("X")), class(&json!("Z")), null]),
            json!([0, 1]),
        ),
        (
            json!([class(&json!("X")), class(&Value::Null), null]),
            json!([0, 1]),
        ),
        (json!([class(&json!("X")), null, null]), json!([0, 1])),
        (json!([class(&json!("X")), null, null]), json!([0, 0])),
    ] {
        let circuit = circuit(&classes, &records, &json!("X"));
        let error = DemBuilder::from_tick_circuit(&circuit, 0.0, 0.0, 0.0, 0.0)
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("D3") && error.contains('X') && error.contains("no common family"),
            "{error}"
        );
    }
}

#[test]
fn cancelled_untyped_and_unclassed_records_do_not_count() {
    for cancelled in [Value::Null, class(&Value::Null)] {
        let circuit = circuit(
            &json!([class(&json!("X")), cancelled, null]),
            &json!([1, 1, 0]),
            &Value::Null,
        );
        let dem = DemBuilder::from_tick_circuit(&circuit, 0.0, 0.0, 0.0, 0.0).unwrap();
        assert_eq!(dem.detectors[0].fault_sector.as_deref(), Some("X"));
    }
}

#[test]
fn explicit_sector_must_agree() {
    let circuit = circuit(
        &json!([class(&json!("Z")), null, null]),
        &json!([0]),
        &json!("X"),
    );
    let error = DemBuilder::from_tick_circuit(&circuit, 0.0, 0.0, 0.0, 0.0)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("D3") && error.contains('X') && error.contains('Z'),
        "{error}"
    );
}

#[test]
fn equal_explicit_sector_is_accepted() {
    let circuit = circuit(
        &json!([class(&json!("Z")), null, null]),
        &json!([0]),
        &json!("Z"),
    );
    let dem = DemBuilder::from_tick_circuit(&circuit, 0.0, 0.0, 0.0, 0.0).unwrap();
    assert_eq!(dem.detectors[0].fault_sector.as_deref(), Some("Z"));
}

#[test]
fn malformed_classes_name_the_ordinal() {
    for classes in [
        json!({}),
        json!([1, null, null]),
        json!([{}, null, null]),
        json!([{"family":"X", "round":0}, null, null]),
        json!([{"block":"", "family":"X", "round":0}, null, null]),
        json!([{"block":7, "family":"X", "round":0}, null, null]),
        json!([{"block":"a", "round":0}, null, null]),
        json!([{"block":"a", "family":"", "round":0}, null, null]),
        json!([{"block":"a", "family":7, "round":0}, null, null]),
        json!([{"block":"a", "family":"X"}, null, null]),
        json!([{"block":"a", "family":"X", "round":0.5}, null, null]),
        json!([{"block":"a", "family":"X", "round":true}, null, null]),
        json!([{"block":"a", "family":"X", "round":null}, null, null]),
        json!([{"block":"a", "family":"X", "round":"0"}, null, null]),
        json!([]),
        json!([null, null, null, null]),
    ] {
        let circuit = circuit(&classes, &json!([0]), &Value::Null);
        let error = DemBuilder::from_tick_circuit(&circuit, 0.0, 0.0, 0.0, 0.0)
            .unwrap_err()
            .to_string();
        let expected = match classes.as_array().map(Vec::len) {
            None => "measurement_classes: expected a JSON list".to_owned(),
            Some(length) if length != 3 => {
                format!("measurement_classes length {length} does not match measurement count 3")
            }
            Some(_) => "ordinal 0".to_owned(),
        };
        assert!(error.contains(&expected), "{classes}: {error}");
    }
}

#[test]
fn manual_classes_follow_map_order_with_nonpositional_ids() {
    // The circuit emits 91 then 17; this hand-made map stores 17 then 91.
    let mut circuit = TickCircuit::new();
    let mut gate = Gate::mz(&[0, 1]);
    gate.meas_ids = vec![MeasId::from_raw(91), MeasId::from_raw(17)].into();
    circuit.tick().try_add_gate(gate).unwrap();
    let emission = pecos_qec::fault_tolerance::circuit_definitions::tick_circuit_emission(&circuit);
    assert_eq!(
        emission,
        vec![Some(MeasId::from_raw(91)), Some(MeasId::from_raw(17))]
    );
    let mut map = DagFaultInfluenceMap::with_capacity(0);
    map.measurements = vec![(1, 1, 0), (0, 0, 0)];
    map.meas_ids = vec![MeasId::from_raw(17), MeasId::from_raw(91)];
    let dem = DemBuilder::new(&map)
        .with_measurement_classes_json(&json!([class(&json!("Z")), class(&json!("X"))]).to_string())
        .unwrap()
        .with_detectors_json(r#"[{"id":0,"meas_ids":[91]},{"id":1,"meas_ids":[17]}]"#)
        .unwrap()
        .try_build()
        .unwrap();
    assert_eq!(dem.detectors[0].fault_sector.as_deref(), Some("X"));
    assert_eq!(dem.detectors[1].fault_sector.as_deref(), Some("Z"));
}

#[test]
fn final_length_and_bad_references_fail_on_both_build_methods() {
    // An empty map keeps a declarative count, so only the class length can
    // disagree; a real map also rejects the changed count itself.
    for (measurements, expected) in [
        (
            vec![],
            "measurement_classes length 1 does not match measurement count 2",
        ),
        (
            vec![(0, 0, 0)],
            "num_measurements=2 disagrees with the 1 measurement(s)",
        ),
    ] {
        let mut map = DagFaultInfluenceMap::with_capacity(0);
        map.measurements = measurements;
        let builder = DemBuilder::new(&map)
            .with_num_measurements(1)
            .with_measurement_classes_json("[null]")
            .unwrap()
            .with_num_measurements(2);
        for result in [builder.build(), builder.try_build()] {
            let error = result.unwrap_err().to_string();
            assert!(error.contains(expected), "{error}");
        }
    }
    let map = DagFaultInfluenceMap::with_capacity(0);
    for (detector, observable, expected) in [
        (
            r#"[{"id":3,"records":[1]}]"#,
            "[]",
            "Detector 3 references record offset 1",
        ),
        (
            r#"[{"id":3,"records":[-2]}]"#,
            "[]",
            "Detector 3 references record offset -2",
        ),
        (
            r#"[{"id":3,"meas_ids":[42]}]"#,
            "[]",
            "Detector 3 references meas_id 42",
        ),
        (
            r#"[{"id":3,"records":[-1]}]"#,
            r#"[{"id":0,"records":[1]}]"#,
            "Observable 0 references record offset 1",
        ),
    ] {
        let builder = DemBuilder::new(&map)
            .with_num_measurements(1)
            .with_measurement_classes_json("[null]")
            .unwrap()
            .with_detectors_json(detector)
            .unwrap()
            .with_observables_json(observable)
            .unwrap();
        for result in [builder.build(), builder.try_build()] {
            let error = result.unwrap_err().to_string();
            assert!(error.contains(expected), "{error}");
        }
    }
}

#[test]
fn coloration_writer_emits_classes_without_detector_sectors() {
    let h = ParityCheckMatrix::from_dense(vec![vec![1, 1]]).unwrap();
    let circuit = coloration_memory_circuit(&h, &h, 2, MemoryBasis::X).unwrap();
    let Attribute::String(json) = circuit.get_meta("measurement_classes").unwrap() else {
        panic!("string attribute")
    };
    let classes: Value = serde_json::from_str(json).unwrap();
    // Coloration measures Z checks then X checks in each syndrome round.
    assert_eq!(
        classes[0],
        json!({"block":"memory", "family":"Z", "round":0})
    );
    assert_eq!(
        classes[1],
        json!({"block":"memory", "family":"X", "round":0})
    );
    assert_eq!(
        classes[4],
        json!({"block":"memory", "family":"X", "round":2})
    );
    let Attribute::String(json) = circuit.get_meta("detectors").unwrap() else {
        panic!("string attribute")
    };
    let detectors: Value = serde_json::from_str(json).unwrap();
    assert!(
        detectors
            .as_array()
            .unwrap()
            .iter()
            .all(|detector| detector.get("fault_sector").is_none())
    );
}

#[test]
fn stamped_id_beyond_the_records_is_rejected_with_classes() {
    // A hand-made map whose stamped ids outnumber its records resolves id 91
    // to record 1 of a one-record circuit.
    let mut map = DagFaultInfluenceMap::with_capacity(0);
    map.measurements = vec![(0, 0, 0)];
    map.meas_ids = vec![MeasId::from_raw(17), MeasId::from_raw(91)];
    let builder = DemBuilder::new(&map)
        .with_measurement_classes_json(&json!([class(&json!("X"))]).to_string())
        .unwrap()
        .with_detectors_json(r#"[{"id":0,"meas_ids":[91]}]"#)
        .unwrap();
    for result in [builder.build(), builder.try_build()] {
        let error = result.unwrap_err().to_string();
        assert!(
            error.contains(
                "detector D0 references measurement 1, outside the 1 measurement classes"
            ),
            "{error}"
        );
    }
}
