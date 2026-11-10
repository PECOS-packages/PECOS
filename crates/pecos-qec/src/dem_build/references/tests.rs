// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use super::*;
use crate::fault_tolerance::circuit_definitions::{ResolvedDetector, ResolvedObservable};

fn definitions() -> CircuitDefinitions {
    CircuitDefinitions {
        detectors: [9, 2]
            .map(|id| ResolvedDetector {
                id,
                reference: None,
                coords: None,
                label: None,
                fault_sector: None,
                measurements: vec![0],
            })
            .to_vec(),
        observables: [7, 1]
            .map(|id| ResolvedObservable {
                id,
                reference: None,
                label: None,
                measurements: vec![0],
            })
            .to_vec(),
        num_measurements: 1,
        measurement_classes: None,
    }
}

#[test]
fn references_follow_definition_ids_even_when_definitions_are_reordered() {
    let mut circuit = TickCircuit::new();
    let mut definitions = definitions();
    let references = NoiselessReferences {
        detectors: BTreeMap::from([(2, 0), (9, 1)]),
        observables: BTreeMap::from([(1, 1), (7, 0)]),
    };
    let (detectors, observables) =
        write_references(&mut circuit, &mut definitions, &references, &[5]).unwrap();
    assert_eq!(
        definitions
            .detectors
            .iter()
            .map(|d| (d.id, d.reference))
            .collect::<Vec<_>>(),
        [(9, Some(1)), (2, Some(0))]
    );
    assert_eq!(
        definitions
            .observables
            .iter()
            .map(|d| (d.id, d.reference))
            .collect::<Vec<_>>(),
        [(7, Some(0)), (1, Some(1))]
    );
    assert_eq!(
        detectors,
        json!([
            {"id": 2, "meas_ids": [5], "reference": 0},
            {"id": 9, "meas_ids": [5], "reference": 1},
        ])
    );
    assert_eq!(
        observables,
        json!([
            {"id": 1, "meas_ids": [5], "reference": 1},
            {"id": 7, "meas_ids": [5], "reference": 0},
        ])
    );
}

#[test]
fn missing_references_cannot_silently_truncate_definitions() {
    for (detectors, observables, missing) in [
        (
            BTreeMap::from([(2, 0)]),
            BTreeMap::from([(1, 1), (7, 0)]),
            "detector 9",
        ),
        (
            BTreeMap::from([(2, 0), (9, 1)]),
            BTreeMap::from([(1, 1)]),
            "observable 7",
        ),
    ] {
        let error = write_references(
            &mut TickCircuit::new(),
            &mut definitions(),
            &NoiselessReferences {
                detectors,
                observables,
            },
            &[5],
        )
        .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("missing noiseless reference for {missing}")
        );
    }
}
