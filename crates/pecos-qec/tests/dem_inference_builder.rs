// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use pecos_qec::{
    DemBuildError, DemBuildInput, DemDefinition, DemInferenceData, DemInferenceSettings,
    DemProgram, DemProgramRequest, DetectorErrorModel, NoiseConfig,
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

#[test]
fn mixed_definitions_share_one_schema_and_ignore_unused_probe_roles() {
    let mut data = probes();
    // Explicit observables must not inspect or fit these unused values.
    data.observable_rows = vec![vec![99]];
    let build = DetectorErrorModel::builder()
        .with_circuit(circuit())
        .with_detector_definitions(vec![
            DemDefinition::explicit(0, vec![0]),
            DemDefinition::inferred(1, "parities", 2),
        ])
        .with_observable_definitions(vec![DemDefinition::explicit(70, vec![1])])
        .with_inference(DemInferenceSettings::default().with_validation_rows(2))
        .with_inference_data(data)
        .build()
        .unwrap();
    let explicit = DetectorErrorModel::builder()
        .with_circuit(circuit())
        .with_detectors(vec![vec![0], vec![0, 1]])
        .with_observables_json(r#"[{"id":70,"meas_ids":[1]}]"#)
        .build()
        .unwrap();
    assert_eq!(build.dem.to_string(), explicit.dem.to_string());
    assert_eq!(
        build.evaluate_runtime_record(&[0, 1]).unwrap(),
        explicit.evaluate_runtime_record(&[0, 1]).unwrap()
    );
    assert_eq!(
        build.inference.as_ref().unwrap().observable_supports,
        Vec::<Vec<usize>>::new()
    );
    assert_eq!(build.definition_origins.len(), 3);
}

struct ExplicitProgram;
impl DemProgram for ExplicitProgram {
    fn prepare(&mut self, request: &DemProgramRequest<'_>) -> Result<DemBuildInput, DemBuildError> {
        assert!(
            request.inference.is_none(),
            "explicit definitions must not request probes"
        );
        assert_eq!(request.detector_outputs, Some([].as_slice()));
        assert_eq!(request.observable_outputs, Some([].as_slice()));
        Ok(DemBuildInput {
            circuit: circuit(),
            probes: None,
        })
    }
}

#[test]
fn explicit_program_skips_inference_even_with_settings() {
    let build = DetectorErrorModel::builder()
        .with_program(ExplicitProgram)
        .with_qubits(2)
        .with_inference(DemInferenceSettings::default())
        .with_detectors(vec![vec![0]])
        .with_observables(vec![])
        .build()
        .unwrap();
    assert!(build.inference.is_none());
}

struct SelectiveProgram;
impl DemProgram for SelectiveProgram {
    fn prepare(&mut self, request: &DemProgramRequest<'_>) -> Result<DemBuildInput, DemBuildError> {
        assert!(request.inference.is_some());
        let outputs = request.detector_outputs.unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].tag, "selected");
        assert_eq!(outputs[0].element, 3);
        assert_eq!(request.observable_outputs.unwrap(), []);
        let mut data = probes();
        // Enough probes for default inference settings, supplied without a
        // with_inference() call: an inferred definition activates the defaults.
        data.raw_rows = data.raw_rows.iter().cycle().take(64).cloned().collect();
        data.detector_rows = data
            .detector_rows
            .iter()
            .cycle()
            .take(64)
            .cloned()
            .collect();
        Ok(DemBuildInput {
            circuit: circuit(),
            probes: Some(data),
        })
    }
}

#[test]
fn adapter_receives_selected_outputs_and_default_settings() {
    let build = DetectorErrorModel::builder()
        .with_program(SelectiveProgram)
        .with_qubits(2)
        .with_detector_definitions(vec![DemDefinition::inferred(0, "selected", 3)])
        .with_observables(vec![])
        .build()
        .unwrap();
    assert_eq!(build.evaluate_runtime_record(&[1, 0]).unwrap().0, vec![1]);
}

#[test]
fn invalid_mixed_definition_ids_fail_before_runtime() {
    assert!(
        DetectorErrorModel::builder()
            .with_program(ExplicitProgram)
            .with_qubits(2)
            .with_detector_definitions(vec![
                DemDefinition::explicit(0, vec![0]),
                DemDefinition::inferred(0, "tag", 0)
            ])
            .build()
            .is_err()
    );
}

#[test]
fn explicit_detector_role_and_inferred_observables_use_only_observable_probes() {
    let mut data = probes();
    data.detector_rows.clear();
    let build = DetectorErrorModel::builder()
        .with_circuit(circuit())
        .with_inference(DemInferenceSettings::default().with_validation_rows(2))
        .with_detectors(vec![vec![1]])
        .with_inference_data(data)
        .build()
        .unwrap();
    assert_eq!(
        build.inference.as_ref().unwrap().detector_supports,
        Vec::<Vec<usize>>::new()
    );
    let (events, obs) = build.evaluate_runtime_record(&[1, 0]).unwrap();
    assert_eq!(events, vec![0]);
    assert!(obs.get(0));
}

#[test]
fn selected_probe_widths_and_repeated_role_setters_are_rejected() {
    assert!(
        DetectorErrorModel::builder()
            .with_circuit(circuit())
            .with_detectors(vec![vec![0]])
            .with_detector_definitions(vec![DemDefinition::inferred(0, "tag", 0)])
            .build()
            .is_err()
    );
    let mut data = probes();
    data.detector_rows[0].push(1);
    assert!(
        DetectorErrorModel::builder()
            .with_circuit(circuit())
            .with_detector_definitions(vec![DemDefinition::inferred(0, "tag", 0)])
            .with_inference(DemInferenceSettings::default().with_validation_rows(2))
            .with_observables(vec![])
            .with_inference_data(data)
            .build()
            .is_err()
    );
}
