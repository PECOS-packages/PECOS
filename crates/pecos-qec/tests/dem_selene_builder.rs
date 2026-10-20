// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

#![cfg(feature = "selene")]

use pecos_programs::Qis;
use pecos_qec::{
    DemBuildError, DemDefinition, DemInferenceError, DemInferenceSettings, DetectorErrorModel,
    NoiseConfig, SeleneDemProgram,
};
use pecos_qis::{SeleneRuntime, selene_simple_runtime};
use pecos_quantum::Attribute;

fn runtime() -> SeleneRuntime {
    selene_simple_runtime().expect("Selene simple runtime is required")
}

fn program(ir: &str) -> SeleneDemProgram {
    SeleneDemProgram::new(Qis::from_string(ir)).with_runtime(runtime())
}

const PARITIES: &str = include_str!("fixtures/dem_programs/parities.ll");

#[test]
fn qis_inference_matches_explicit_definitions() {
    let noise = NoiseConfig::new(0.001, 0.01, 0.02, 0.001);
    let inferred = DetectorErrorModel::builder()
        .with_program(program(PARITIES))
        .with_qubits(3)
        .with_seed(42)
        .with_noise(noise.clone())
        .with_idle_after_2q(1.0)
        .with_inference(DemInferenceSettings::default())
        .build()
        .unwrap();
    let report = inferred.inference.as_ref().unwrap();
    assert_eq!(report.raw_measurement_ids, [2, 0, 1]);
    assert_eq!(report.detector_supports, [vec![2, 0]]);
    assert_eq!(report.observable_supports, [vec![1]]);
    let explicit = DetectorErrorModel::builder()
        .with_program(program(PARITIES))
        .with_qubits(3)
        .with_noise(noise)
        .with_idle_after_2q(1.0)
        .with_detectors(vec![vec![2, 0]])
        .with_observables(vec![vec![1]])
        .build()
        .unwrap();
    assert!(inferred.dem.to_string().contains("error("));
    assert_eq!(inferred.dem.to_string(), explicit.dem.to_string());
    for record in [[0, 0, 0], [1, 0, 1], [1, 1, 0]] {
        assert_eq!(
            inferred.evaluate_runtime_record(&record).unwrap(),
            explicit.evaluate_runtime_record(&record).unwrap()
        );
    }
    assert_eq!(
        inferred.circuit.get_meta("dem_raw_measurement_binding"),
        Some(&Attribute::String("probe_correlated_result_ids".into()))
    );
}

#[test]
fn mixed_definitions_fit_only_requested_outputs() {
    let build = DetectorErrorModel::builder()
        .with_program(program(PARITIES))
        .with_qubits(3)
        .with_detector_definitions(vec![
            DemDefinition::explicit(0, vec![0]),
            DemDefinition::inferred(1, "DETECTOR", 0),
        ])
        .with_observable_definitions(vec![DemDefinition::explicit(70, vec![1])])
        .build()
        .unwrap();
    assert_eq!(
        build.inference.as_ref().unwrap().detector_supports,
        [vec![2, 0]]
    );
    let (detectors, observables) = build.evaluate_runtime_record(&[1, 1, 0]).unwrap();
    assert_eq!(detectors, [1, 1]);
    assert!(observables.get(70));
}

#[test]
fn requested_nonlinear_output_is_rejected() {
    let error = DetectorErrorModel::builder()
        .with_program(program(PARITIES))
        .with_qubits(3)
        .with_detector_definitions(vec![DemDefinition::inferred(0, "unused", 0)])
        .build()
        .err()
        .unwrap();
    assert!(
        matches!(
            error,
            DemBuildError::Inference(DemInferenceError::NonAffine)
        ),
        "{error}"
    );
}

#[test]
fn explicit_program_does_not_require_inference_outputs() {
    let build = DetectorErrorModel::builder()
        .with_program(program(include_str!(
            "fixtures/dem_programs/explicit_only.ll"
        )))
        .with_qubits(1)
        .with_detectors(vec![vec![0]])
        .with_observables(vec![])
        .with_inference(DemInferenceSettings::default())
        .build()
        .unwrap();
    assert!(build.inference.is_none());
    assert!(
        build
            .circuit
            .get_meta("dem_raw_measurement_binding")
            .is_none()
    );
    assert_eq!(
        build.circuit.get_meta("dem_schedule_validation"),
        Some(&Attribute::String("empirical_trace_comparison".into()))
    );
}

#[test]
fn measurement_dependent_schedule_is_rejected() {
    for inferred in [false, true] {
        let builder = DetectorErrorModel::builder()
            .with_program(program(include_str!(
                "fixtures/dem_programs/changing_schedule.ll"
            )))
            .with_qubits(2);
        let builder = if inferred {
            builder.with_inference(DemInferenceSettings::default())
        } else {
            builder
                .with_detectors(vec![vec![0, 1]])
                .with_observables(vec![])
        };
        let error = builder.build().err().unwrap();
        assert!(error.to_string().contains("schedule changed"), "{error}");
    }
}

#[test]
#[cfg(feature = "hugr")]
fn hugr_compiles_and_infers_without_python() {
    let build = DetectorErrorModel::builder()
        .with_program(
            SeleneDemProgram::from_hugr(
                include_bytes!("fixtures/dem_programs/parities.hugr").to_vec(),
            )
            .with_runtime(runtime()),
        )
        .with_qubits(3)
        .with_inference(DemInferenceSettings::default())
        .build()
        .unwrap();
    let report = build.inference.unwrap();
    assert_eq!(report.raw_measurement_ids, [2, 0, 1]);
    assert_eq!(report.detector_supports, [vec![2, 0]]);
    assert_eq!(report.observable_supports, [vec![1]]);
}

#[test]
fn direct_result_ids_skip_correlation_and_preserve_multiple_output_tags() {
    let build = DetectorErrorModel::builder()
        .with_program(
            program(include_str!("fixtures/dem_programs/direct_outputs.ll"))
                .with_schedule_validation_shots(0),
        )
        .with_qubits(3)
        .with_inference(
            DemInferenceSettings::default()
                .with_raw_tag("physical")
                .with_detector_tag("checks")
                .with_observable_tags(["logical_z", "logical_x"])
                .with_probe_shots(40)
                .with_validation_rows(8)
                .with_provenance_shots(100),
        )
        .build()
        .unwrap();
    let report = build.inference.unwrap();
    assert_eq!(report.raw_measurement_ids, [0, 1, 2]);
    assert_eq!(report.detector_supports, [vec![0, 1]]);
    assert_eq!(report.observable_supports, [vec![0], vec![1], vec![2]]);
    assert_eq!(
        build.circuit.get_meta("dem_raw_measurement_binding"),
        Some(&Attribute::String("runtime_result_ids".into()))
    );
    assert_eq!(
        build.circuit.get_meta("dem_schedule_validation_shots"),
        Some(&Attribute::String("40".into()))
    );
}
