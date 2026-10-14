// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

use pecos_core::{MeasId, QubitId};
use pecos_qec::fault_tolerance::fault_sampler::{FaultCatalog, FaultCatalogError, FaultChannel};
use pecos_quantum::{Attribute, Gate, TickCircuit};

fn circuit() -> TickCircuit {
    let mut tc = TickCircuit::new();
    let mut gate = Gate::mz(&[1, 0]);
    gate.meas_ids
        .extend([MeasId::from_raw(17), MeasId::from_raw(9)]);
    tc.tick().try_add_gate(gate).unwrap();
    tc
}

fn metadata(tc: &mut TickCircuit, kind: &str, json: &str) {
    tc.set_meta(kind, Attribute::String(json.into()));
}

fn effects(tc: &TickCircuit) -> Vec<(Vec<usize>, Vec<usize>)> {
    FaultCatalog::from_circuit(tc)
        .unwrap()
        .locations
        .into_iter()
        .filter(|loc| loc.channel == FaultChannel::PMeas)
        .map(|loc| {
            let fault = &loc.faults[0];
            (
                fault.affected_detectors.clone(),
                fault.affected_observables.clone(),
            )
        })
        .collect()
}

#[test]
fn absolute_records_name_first_emission() {
    let mut tc = circuit();
    for kind in ["detectors", "observables"] {
        metadata(&mut tc, kind, r#"[{"id":0,"records":[0]}]"#);
    }
    assert_eq!(effects(&tc), vec![(vec![0], vec![0]), (vec![], vec![])]);
}

#[test]
fn measurement_ids_and_annotations_name_first_emission() {
    for source in ["records", "meas_ids", "annotations"] {
        let mut tc = circuit();
        if source == "annotations" {
            let first = tc.meas_ref(0, 0, QubitId(1)).unwrap();
            tc.detector(&[first]).unwrap();
            tc.observable(&[first]).unwrap();
        } else {
            let value = if source == "records" { -2 } else { 17 };
            for kind in ["detectors", "observables"] {
                metadata(
                    &mut tc,
                    kind,
                    &format!(r#"[{{"id":0,"{source}":[{value}]}}]"#),
                );
            }
        }
        assert_eq!(effects(&tc), vec![(vec![0], vec![0]), (vec![], vec![])]);
    }
}

#[test]
fn invalid_catalog_definitions_are_errors() {
    for kind in ["detectors", "observables"] {
        for json in [
            r#"[{"id":0,"records":[2]}]"#,
            r#"[{"id":0,"records":[-3]}]"#,
            r#"[{"id":0,"meas_ids":[0]}]"#,
            r#"[{"id":0,"records":[-1]}"#,
            r#"[{"id":0,"records":[0]},{"id":0,"records":[1]}]"#,
        ] {
            let mut tc = circuit();
            metadata(&mut tc, kind, json);
            assert!(
                matches!(
                    FaultCatalog::from_circuit(&tc),
                    Err(FaultCatalogError::Definition(_))
                ),
                "{kind}: {json}"
            );
        }
    }
    for kind in ["detectors", "observables", "num_measurements"] {
        let mut tc = circuit();
        tc.set_meta(kind, Attribute::Int(2));
        assert!(matches!(
            FaultCatalog::from_circuit(&tc),
            Err(FaultCatalogError::Definition(_))
        ));
    }
}

#[test]
fn measurement_count_must_match_emission_count() {
    let mut tc = circuit();
    metadata(&mut tc, "num_measurements", "2");
    assert!(FaultCatalog::from_circuit(&tc).is_ok());
    for value in ["1", "3", "invalid"] {
        metadata(&mut tc, "num_measurements", value);
        assert!(matches!(
            FaultCatalog::from_circuit(&tc),
            Err(FaultCatalogError::Definition(_))
        ));
    }
}

#[test]
fn out_of_order_definition_ids_keep_their_effects() {
    let mut tc = circuit();
    for kind in ["detectors", "observables"] {
        metadata(
            &mut tc,
            kind,
            r#"[{"id":1,"records":[-2]},{"id":0,"records":[-1]}]"#,
        );
    }
    assert_eq!(effects(&tc), vec![(vec![1], vec![1]), (vec![0], vec![0])]);
}

#[test]
fn metadata_and_annotations_must_agree() {
    for kind in ["detectors", "observables"] {
        let mut tc = circuit();
        let first = tc.meas_ref(0, 0, QubitId(1)).unwrap();
        if kind == "detectors" {
            tc.detector(&[first]).unwrap();
        } else {
            tc.observable(&[first]).unwrap();
        }
        for json in ["", "[]", r#"[{"id":0,"records":[0]}]"#] {
            metadata(&mut tc, kind, json);
            let expected = if kind == "detectors" {
                (vec![0], vec![])
            } else {
                (vec![], vec![0])
            };
            assert_eq!(effects(&tc), vec![expected, (vec![], vec![])]);
        }
        metadata(&mut tc, kind, r#"[{"id":0,"records":[1]}]"#);
        assert!(matches!(
            FaultCatalog::from_circuit(&tc),
            Err(FaultCatalogError::Definition(_))
        ));
    }
}

#[test]
fn measurement_walks_follow_tick_batch_instance_and_qubit_order() {
    use pecos_core::gate_type::GateType;

    let mut tc = TickCircuit::new();
    for types in [
        [GateType::MX, GateType::MZ],
        [GateType::MeasureFree, GateType::MPZ],
    ] {
        let mut tick = tc.tick();
        for (kind, qubits) in types.into_iter().zip([[3, 1], [2, 0]]) {
            let mut gate = Gate::mz(&qubits);
            gate.gate_type = kind;
            tick.try_add_gate(gate).unwrap();
        }
    }
    let mut leaked = Gate::mz(&[4]);
    leaked.gate_type = GateType::MeasureLeaked;
    tc.tick().try_add_gate(leaked).unwrap();
    metadata(&mut tc, "num_measurements", "8");
    metadata(
        &mut tc,
        "detectors",
        &format!(
            "[{}]",
            (0..8)
                .map(|i| format!(r#"{{"id":{i},"records":[{i}]}}"#))
                .collect::<Vec<_>>()
                .join(",")
        ),
    );
    let catalog = FaultCatalog::from_circuit(&tc).unwrap();
    let measurements: Vec<_> = catalog
        .locations
        .iter()
        .filter(|loc| loc.channel == FaultChannel::PMeas)
        .collect();
    assert_eq!(
        measurements
            .iter()
            .map(|loc| loc.qubits[0])
            .collect::<Vec<_>>(),
        vec![3, 1, 2, 0, 3, 1, 2, 0]
    );
    for (position, location) in measurements.iter().enumerate() {
        assert_eq!(location.faults[0].affected_detectors, vec![position]);
    }
}
