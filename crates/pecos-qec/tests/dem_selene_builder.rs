// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

#![cfg(feature = "selene")]

use pecos_programs::Qis;
use pecos_qec::{
    DemAnnotationOrigin, DemBuildError, DemDefinition, DemInferenceError, DemInferenceSettings,
    DemOutputSelector, DetectorErrorModel, NoiseConfig, SeleneDemProgram,
};
use pecos_qis::{SeleneRuntime, selene_simple_runtime, selene_soft_rz_runtime};
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
fn public_runtimes_support_explicit_and_inferred_lazy_measurement_dems() {
    for runtime in [selene_simple_runtime, selene_soft_rz_runtime] {
        for inferred in [false, true] {
            let source = if inferred {
                PARITIES
            } else {
                include_str!("fixtures/dem_programs/explicit_only.ll")
            };
            let program = SeleneDemProgram::new(Qis::from_string(source))
                .with_runtime(runtime().expect("Selene public runtime is required"));
            let builder = DetectorErrorModel::builder()
                .with_program(program)
                .with_qubits(if inferred { 3 } else { 1 });
            let build = if inferred {
                builder.with_inference(DemInferenceSettings::default())
            } else {
                builder
                    .with_detectors(vec![vec![0]])
                    .with_observables(vec![])
            }
            .build()
            .unwrap();
            if inferred {
                let report = build.inference.as_ref().unwrap();
                assert_eq!(report.raw_measurement_ids, [2, 0, 1]);
                assert_eq!(report.detector_supports, [vec![2, 0]]);
                assert_eq!(report.observable_supports, [vec![1]]);
                let (detectors, observables) = build.evaluate_runtime_record(&[1, 1, 0]).unwrap();
                assert_eq!(detectors, [1]);
                assert!(observables.get(0));
            } else {
                assert!(build.inference.is_none());
                let (detectors, observables) = build.evaluate_runtime_record(&[1]).unwrap();
                assert_eq!(detectors, [1]);
                assert!(observables.is_zero());
            }
        }
    }
}

#[test]
fn explicit_program_requires_positive_schedule_validation_shots() {
    let error = DetectorErrorModel::builder()
        .with_program(
            program(include_str!("fixtures/dem_programs/explicit_only.ll"))
                .with_schedule_validation_shots(0),
        )
        .with_qubits(1)
        .with_detectors(vec![vec![0]])
        .with_observables(vec![])
        .build()
        .err()
        .expect("zero schedule-validation shots must be rejected")
        .to_string();
    assert!(
        error.contains("schedule validation shots must be positive"),
        "{error}"
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
    let expected = [
        ("detector", "checks", 0),
        ("observable", "logical_z", 0),
        ("observable", "logical_x", 0),
        ("observable", "logical_x", 1),
    ];
    assert_eq!(build.definition_origins.len(), expected.len());
    for (origin, (role, tag, element)) in build.definition_origins.iter().zip(expected) {
        assert_eq!(origin.role, role);
        assert_eq!(
            origin.source,
            DemAnnotationOrigin::Inferred(DemOutputSelector {
                tag: tag.into(),
                element
            }),
        );
    }
    for (key, expected) in [
        (
            "detectors",
            serde_json::json!([
                {"id": 0, "meas_ids": [0, 1], "inferred_from_result_tag": "checks"},
            ]),
        ),
        (
            "observables",
            serde_json::json!([
                {"id": 0, "meas_ids": [0], "inferred_from_result_tag": "logical_z", "result_element": 0},
                {"id": 1, "meas_ids": [1], "inferred_from_result_tag": "logical_x", "result_element": 0},
                {"id": 2, "meas_ids": [2], "inferred_from_result_tag": "logical_x", "result_element": 1},
            ]),
        ),
    ] {
        let Some(Attribute::String(json)) = build.circuit.get_meta(key) else {
            panic!("missing {key} metadata");
        };
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(json).unwrap(),
            expected
        );
    }

    assert_eq!(
        build.circuit.get_meta("dem_raw_measurement_binding"),
        Some(&Attribute::String("runtime_result_ids".into()))
    );
    assert_eq!(
        build.circuit.get_meta("dem_schedule_validation_shots"),
        Some(&Attribute::String("40".into()))
    );
}

#[test]
fn integer_output_positions_follow_complete_stream() {
    const INTEGER_OUTPUTS: &str = include_str!("fixtures/dem_programs/integer_outputs.ll");
    for (tag, element, expected) in [
        ("checks", 0, 0),
        ("checks", 1, 1),
        ("bits", 0, 1),
        ("bits", 1, 2),
        ("counts", 1, 2),
    ] {
        let build = DetectorErrorModel::builder()
            .with_program(program(INTEGER_OUTPUTS))
            .with_qubits(3)
            .with_detector_definitions(vec![DemDefinition::inferred(0, tag, element)])
            .build()
            .unwrap();
        let report = build.inference.unwrap();
        assert_eq!(report.raw_measurement_ids, [0, 1, 2]);
        assert_eq!(
            report.detector_supports,
            [vec![expected]],
            "{tag}[{element}]"
        );
    }
    let error = DetectorErrorModel::builder()
        .with_program(program(INTEGER_OUTPUTS))
        .with_qubits(3)
        .with_detector_definitions(vec![DemDefinition::inferred(0, "counts", 0)])
        .build()
        .err()
        .unwrap()
        .to_string();
    assert!(
        error.contains("counts") && error.contains("element 0") && error.contains("value 2"),
        "{error}"
    );
    let build = DetectorErrorModel::builder()
        .with_program(program(INTEGER_OUTPUTS))
        .with_qubits(3)
        .with_inference(DemInferenceSettings::default().with_detector_tag("checks"))
        .with_observables(vec![])
        .build()
        .unwrap();
    assert_eq!(
        build.inference.unwrap().detector_supports,
        [vec![0], vec![1]]
    );
}

#[test]
fn complete_named_results_are_isolated_per_shot() {
    use pecos_engines::{coin_toss, sim_builder};
    use pecos_qis::{OperationTraceStore, helios_interface_builder, qis_engine};
    use pecos_qis_ffi_types::NamedResult;
    use std::sync::Arc;
    let store = OperationTraceStore::default();
    sim_builder()
        .classical(
            qis_engine()
                .runtime(runtime())
                .interface(helios_interface_builder())
                .program_ir(include_str!("fixtures/dem_programs/integer_outputs.ll").to_string())
                .trace_operations_in_memory_to(Arc::clone(&store)),
        )
        .quantum(coin_toss())
        .qubits(3)
        .workers(1)
        .seed(42)
        .run(4)
        .unwrap();
    let chunks = store.lock().unwrap();
    let named: Vec<_> = chunks
        .iter()
        .filter(|chunk| chunk.stage == "named_results")
        .collect();
    assert_eq!(named.len(), 4);
    for chunk in named {
        assert_eq!(chunk.named_results["checks"].len(), 2);
        assert_eq!(chunk.named_results["bits"].len(), 2);
        assert_eq!(chunk.named_results["raw measurements"].len(), 3);
        let NamedResult::I64(counts) = &chunk.named_results["counts"] else {
            panic!("integer counts");
        };
        assert_eq!(counts.len(), 2);
        assert_eq!(counts[0], 2);
    }
    assert!(
        chunks
            .iter()
            .filter(|chunk| chunk.stage != "named_results")
            .all(|chunk| chunk.named_results.is_empty())
    );
}

#[test]
fn integer_outputs_bind_raw_ids_by_probe_correlation() {
    // The program reads every future before emitting raw outputs, so the
    // runtime cannot attach direct IDs; correlation recovers them.
    let build = DetectorErrorModel::builder()
        .with_program(program(include_str!(
            "fixtures/dem_programs/integer_outputs.ll"
        )))
        .with_qubits(3)
        .with_detector_definitions(vec![DemDefinition::inferred(0, "checks", 0)])
        .build()
        .unwrap();
    assert_eq!(
        build.circuit.get_meta("dem_raw_measurement_binding"),
        Some(&Attribute::String("probe_correlated_result_ids".into()))
    );
}

#[test]
fn classical_delay_trace_uses_existing_idle_noise_in_nanoseconds() {
    use pecos_core::gate_type::GateType;
    use pecos_qec::IdleNoiseFamily;
    use std::collections::BTreeMap;

    let source = include_str!("fixtures/dem_programs/explicit_only.ll").replace(
        "  %lazy_measure =",
        "  %delay = call i32 @selene_simulate_delay(ptr null, i64 101)\n  %lazy_measure =",
    );
    let source = format!("{source}\ndeclare i32 @selene_simulate_delay(ptr, i64)\n");
    let rate_per_nanosecond = 0.0001;
    let build = DetectorErrorModel::builder()
        .with_program(program(&source))
        .with_qubits(1)
        .with_detectors(vec![vec![0]])
        .with_observables(vec![])
        .with_noise(
            NoiseConfig::new(0.0, 0.0, 0.0, 0.0).set_idle_linear(IdleNoiseFamily::new(
                rate_per_nanosecond,
                BTreeMap::from([("X".into(), 1.0)]),
            )),
        )
        .build()
        .unwrap();
    let idles: Vec<_> = build
        .circuit
        .iter_gate_batches()
        .filter(|gate| gate.as_gate().gate_type == GateType::Idle)
        .map(|gate| gate.as_gate().params[0])
        .collect();
    assert_eq!(idles, [101.0]);
    let (mechanisms, _) = build.dem.to_mechanisms();
    assert_eq!(mechanisms.len(), 1);
    let (probability, detectors, observables) = &mechanisms[0];
    assert!((probability - 101.0 * rate_per_nanosecond).abs() < 1e-12);
    assert_eq!(detectors, &[0]);
    assert_eq!(observables.as_slice(), [] as [u32; 0]);
}
