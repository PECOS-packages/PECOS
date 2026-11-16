// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use pecos_qec::{
    DemBuildError, DemBuildInput, DemInferenceData, DemInferenceSettings, DemProgram,
    DemProgramRequest, DetectorErrorModel, NoiseConfig,
};
use pecos_quantum::{TickCircuit, pass};
use std::collections::BTreeMap;

fn circuit() -> TickCircuit {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0, 1]);
    circuit.tick().cx(&[(0, 1)]);
    circuit.tick().mz(&[0, 1]);
    pass::assign_missing_meas_ids(&mut circuit);
    circuit
}

fn probes() -> DemInferenceData {
    let raw_rows: Vec<Vec<u8>> = (0..8).map(|i| vec![i & 1, (i >> 1) & 1]).collect();
    DemInferenceData {
        raw_measurement_ids: vec![1, 0],
        detector_rows: raw_rows.iter().map(|row| vec![row[0] ^ row[1]]).collect(),
        observable_rows: raw_rows.iter().map(|row| vec![row[1]]).collect(),
        raw_rows,
    }
}

#[test]
fn inferred_and_explicit_builds_share_noise_idle_and_evaluation() {
    let noise = NoiseConfig::new(0.001, 0.01, 0.02, 0.001);
    let explicit = DetectorErrorModel::builder()
        .with_circuit(circuit())
        .with_detectors(vec![vec![0, 1]])
        .with_observables(vec![vec![0]])
        .with_noise(noise.clone())
        .with_idle_after_2q(1.0)
        .build()
        .unwrap();
    let inferred = DetectorErrorModel::builder()
        .with_circuit(circuit())
        .with_inference(DemInferenceSettings::default().with_validation_rows(2))
        .with_inference_data(probes())
        .with_noise(noise)
        .with_idle_after_2q(1.0)
        .build()
        .unwrap();
    assert_eq!(explicit.dem.to_string(), inferred.dem.to_string());
    assert_eq!(
        explicit.evaluate_runtime_record(&[1, 0]).unwrap(),
        inferred.evaluate_runtime_record(&[1, 0]).unwrap()
    );
    let columns = BTreeMap::from([("raw measurements".into(), vec![vec![0, 1], vec![1, 1]])]);
    let rows = inferred.evaluate_result_columns(&columns).unwrap();
    assert_eq!(rows[0].0, vec![1]);
    assert!(rows[0].1.get(0));
    assert_eq!(rows[1].0, vec![0]);
    assert!(rows[1].1.get(0));
}

#[test]
fn conflicts_and_bad_ids_fail_before_building() {
    for reverse in [false, true] {
        let builder = DetectorErrorModel::builder().with_circuit(circuit());
        let builder = if reverse {
            builder
                .with_detectors(vec![vec![0]])
                .with_inference(DemInferenceSettings::default())
        } else {
            builder
                .with_inference(DemInferenceSettings::default())
                .with_detectors(vec![vec![0]])
        };
        assert!(builder.build().is_err());
    }
    let mut data = probes();
    data.raw_measurement_ids[0] = 9;
    assert!(
        DetectorErrorModel::builder()
            .with_circuit(circuit())
            .with_inference(DemInferenceSettings::default().with_validation_rows(2))
            .with_inference_data(data)
            .build()
            .is_err()
    );
}

struct PreparedProgram;
impl DemProgram for PreparedProgram {
    fn prepare(&mut self, request: &DemProgramRequest<'_>) -> Result<DemBuildInput, DemBuildError> {
        assert_eq!(request.num_qubits, 2);
        assert_eq!(request.seed, 7);
        assert!(request.inference.is_some());
        Ok(DemBuildInput {
            circuit: circuit(),
            probes: Some(probes()),
        })
    }
}

#[test]
fn program_adapter_receives_shared_execution_and_inference_settings() {
    let build = DetectorErrorModel::builder()
        .with_program(PreparedProgram)
        .with_qubits(2)
        .with_seed(7)
        .with_inference(DemInferenceSettings::default().with_validation_rows(2))
        .build()
        .unwrap();
    assert_eq!(build.dem.num_detectors(), 1);
    assert!(build.inference.is_some());
}

#[test]
fn result_evaluation_preserves_wide_observable_ids() {
    let build = DetectorErrorModel::builder()
        .with_circuit(circuit())
        .with_detectors(vec![vec![0]])
        .with_observables_json(r#"[{"id":70,"meas_ids":[1]}]"#)
        .build()
        .unwrap();
    assert!(build.evaluate_runtime_record(&[0, 1]).unwrap().1.get(70));
}
