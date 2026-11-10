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
#[cfg(feature = "hugr")]
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
    let audit = inferred.prepared().audit();
    assert_eq!(audit.named_result_binding, "probe_correlated_result_ids");
    let inference_audit = audit.inference.unwrap();
    assert_eq!(inference_audit.provenance_shots, Some(32));
    assert_eq!(
        inference_audit.schedule_validation.as_deref(),
        Some("empirical_trace_comparison")
    );
    let report = inferred.prepared().inference().unwrap();
    assert_eq!(report.raw_measurement_ids, [2, 0, 1]);
    assert_eq!(report.detector_supports, [vec![2, 0]]);
    assert_eq!(report.observable_supports, [vec![1]]);
    let explicit = DetectorErrorModel::builder()
        .with_program(hugr_program(include_bytes!(
            "fixtures/dem_programs/parities.hugr"
        )))
        .with_qubits(3)
        .with_noise(noise)
        .with_idle_after_2q(1.0)
        .with_detectors(vec![vec![2, 0]])
        .with_observables(vec![vec![1]])
        .build()
        .unwrap();
    assert!(inferred.dem().to_string().contains("error("));
    assert_eq!(inferred.dem().to_string(), explicit.dem().to_string());
    for record in [[0, 0, 0], [1, 0, 1], [1, 1, 0]] {
        assert_eq!(
            inferred
                .prepared()
                .evaluate_runtime_record(&record)
                .unwrap(),
            explicit
                .prepared()
                .evaluate_runtime_record(&record)
                .unwrap()
        );
    }
    assert_eq!(
        inferred
            .prepared()
            .circuit()
            .get_meta("dem_raw_measurement_binding"),
        Some(&Attribute::String("probe_correlated_result_ids".into()))
    );
}

#[test]
fn mixed_definitions_fit_only_requested_outputs() {
    let build = DetectorErrorModel::builder()
        .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
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
        build.prepared().inference().unwrap().detector_supports,
        [vec![2, 0]]
    );
    let (detectors, observables) = build
        .prepared()
        .evaluate_runtime_record(&[1, 1, 0])
        .unwrap();
    assert_eq!(detectors, [1, 1]);
    assert!(observables.get(70));
}

#[test]
fn requested_nonlinear_output_is_rejected() {
    let error = DetectorErrorModel::builder()
        .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
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
fn explicit_qis_is_rejected_before_capture_even_with_inference_settings() {
    // Invalid IR would fail if capture or compilation happened before policy validation.
    let error = DetectorErrorModel::builder()
        .with_program(SeleneDemProgram::new(Qis::from_string("must not execute")))
        .with_qubits(1)
        .with_detectors(vec![vec![0]])
        .with_observables(vec![])
        .with_inference(DemInferenceSettings::default())
        .prepare()
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("requires static schedule evidence"),
        "{error}"
    );
}

#[cfg(feature = "hugr")]
fn hugr_program(bytes: &[u8]) -> SeleneDemProgram {
    SeleneDemProgram::from_hugr(
        bytes.to_vec(),
        pecos_qec::QSystemPlatform::Helios,
        pecos_qec::OptimizationLevel::Default,
    )
    .with_runtime(runtime())
}

#[test]
#[cfg(feature = "hugr")]
fn explicit_program_does_not_require_inference_outputs() {
    let prepared = DetectorErrorModel::builder()
        .with_program(hugr_program(include_bytes!(
            "fixtures/dem_programs/explicit_only.hugr"
        )))
        .with_qubits(1)
        .with_detectors(vec![vec![0]])
        .with_observables(vec![])
        .with_inference(DemInferenceSettings::default())
        .prepare()
        .unwrap();
    assert!(prepared.inference().is_none());
    let audit = prepared.audit();
    assert_eq!(
        audit.static_schedule_source.as_deref(),
        Some("native_hugr_control_flow")
    );
    assert_eq!(audit.schedule_validation_shots, None);
    assert!(
        prepared
            .circuit()
            .get_meta("dem_schedule_validation")
            .is_none()
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
            let program = if inferred {
                program
            } else {
                program.with_frontend_static_schedule("explicit_only_fixture")
            };
            let builder = DetectorErrorModel::builder()
                .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
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
                let report = build.prepared().inference().unwrap();
                assert_eq!(report.raw_measurement_ids, [2, 0, 1]);
                assert_eq!(report.detector_supports, [vec![2, 0]]);
                assert_eq!(report.observable_supports, [vec![1]]);
                let (detectors, observables) = build
                    .prepared()
                    .evaluate_runtime_record(&[1, 1, 0])
                    .unwrap();
                assert_eq!(detectors, [1]);
                assert!(observables.get(0));
            } else {
                assert!(build.prepared().inference().is_none());
                let (detectors, observables) =
                    build.prepared().evaluate_runtime_record(&[1]).unwrap();
                assert_eq!(detectors, [1]);
                assert!(observables.is_zero());
            }
        }
    }
}

#[test]
fn explicit_program_requires_positive_schedule_validation_shots() {
    let error = DetectorErrorModel::builder()
        .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
        .with_program(
            program(include_str!("fixtures/dem_programs/explicit_only.ll"))
                .with_schedule_validation_shots(0),
        )
        .with_qubits(1)
        .with_detectors(vec![vec![0]])
        .with_observables(vec![])
        .build()
        .err()
        .expect("uncertified QIS must be rejected before shot validation")
        .to_string();
    assert!(
        error.contains("requires static schedule evidence"),
        "{error}"
    );
}

#[test]
fn measurement_dependent_schedule_is_rejected() {
    for inferred in [false, true] {
        let builder = DetectorErrorModel::builder()
            .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
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
        let expected = if inferred {
            "schedule changed"
        } else {
            "requires static schedule evidence"
        };
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[test]
#[cfg(feature = "hugr")]
fn hugr_compiles_and_infers_without_python() {
    let build = DetectorErrorModel::builder()
        .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
        .with_program(
            SeleneDemProgram::from_hugr(
                include_bytes!("fixtures/dem_programs/parities.hugr").to_vec(),
                pecos_qec::QSystemPlatform::Helios,
                pecos_qec::OptimizationLevel::Default,
            )
            .with_runtime(runtime()),
        )
        .with_qubits(3)
        .with_inference(DemInferenceSettings::default())
        .build()
        .unwrap();
    let report = build.prepared().inference().unwrap();
    assert_eq!(report.raw_measurement_ids, [2, 0, 1]);
    assert_eq!(report.detector_supports, [vec![2, 0]]);
    assert_eq!(report.observable_supports, [vec![1]]);
}

#[test]
fn direct_result_ids_skip_correlation_and_preserve_multiple_output_tags() {
    let build = DetectorErrorModel::builder()
        .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
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
    let report = build.prepared().inference().unwrap();
    assert_eq!(report.raw_measurement_ids, [0, 1, 2]);
    assert_eq!(report.detector_supports, [vec![0, 1]]);
    assert_eq!(report.observable_supports, [vec![0], vec![1], vec![2]]);
    let expected = [
        ("detector", "checks", 0),
        ("observable", "logical_z", 0),
        ("observable", "logical_x", 0),
        ("observable", "logical_x", 1),
    ];
    assert_eq!(build.prepared().definition_origins().len(), expected.len());
    for (origin, (role, tag, element)) in build.prepared().definition_origins().iter().zip(expected)
    {
        assert_eq!(origin.role, role);
        assert_eq!(
            origin.source,
            DemAnnotationOrigin::Inferred(DemOutputSelector {
                tag: tag.into(),
                element
            }),
        );
    }
    for (key, mut expected) in [
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
        for entry in expected.as_array_mut().unwrap() {
            entry["reference"] = serde_json::json!(0);
        }
        let Some(Attribute::String(json)) = build.prepared().circuit().get_meta(key) else {
            panic!("missing {key} metadata");
        };
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(json).unwrap(),
            expected
        );
    }

    assert_eq!(
        build
            .prepared()
            .circuit()
            .get_meta("dem_raw_measurement_binding"),
        Some(&Attribute::String("runtime_result_ids".into()))
    );
    assert_eq!(
        build
            .prepared()
            .circuit()
            .get_meta("dem_schedule_validation_shots"),
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
            .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
            .with_program(program(INTEGER_OUTPUTS))
            .with_qubits(3)
            .with_detector_definitions(vec![DemDefinition::inferred(0, tag, element)])
            .build()
            .unwrap();
        let report = build.prepared().inference().unwrap();
        assert_eq!(report.raw_measurement_ids, [0, 1, 2]);
        assert_eq!(
            report.detector_supports,
            [vec![expected]],
            "{tag}[{element}]"
        );
    }
    let error = DetectorErrorModel::builder()
        .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
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
        .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
        .with_program(program(INTEGER_OUTPUTS))
        .with_qubits(3)
        .with_inference(DemInferenceSettings::default().with_detector_tag("checks"))
        .with_observables(vec![])
        .build()
        .unwrap();
    assert_eq!(
        build.prepared().inference().unwrap().detector_supports,
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
        .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
        .with_program(program(include_str!(
            "fixtures/dem_programs/integer_outputs.ll"
        )))
        .with_qubits(3)
        .with_detector_definitions(vec![DemDefinition::inferred(0, "checks", 0)])
        .build()
        .unwrap();
    assert_eq!(
        build
            .prepared()
            .circuit()
            .get_meta("dem_raw_measurement_binding"),
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
        .with_program(program(&source).with_frontend_static_schedule("classical_delay_fixture"))
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
        .prepared()
        .circuit()
        .iter_gate_batches()
        .filter(|gate| gate.as_gate().gate_type == GateType::Idle)
        .map(|gate| gate.as_gate().params[0])
        .collect();
    assert_eq!(idles, [101.0]);
    let (mechanisms, _) = build.dem().to_mechanisms();
    assert_eq!(mechanisms.len(), 1);
    let (probability, detectors, observables) = &mechanisms[0];
    assert!((probability - 101.0 * rate_per_nanosecond).abs() < 1e-12);
    assert_eq!(detectors, &[0]);
    assert_eq!(observables.as_slice(), [] as [u32; 0]);
}

#[test]
fn half_turn_keeps_two_qubit_noise_and_idle_sites() {
    use pecos_core::gate_type::GateType;
    for runtime in [selene_simple_runtime, selene_soft_rz_runtime] {
        let program = SeleneDemProgram::new(Qis::from_string(include_str!(
            "fixtures/dem_programs/half_turn.ll"
        )))
        .with_runtime(runtime().expect("Selene public runtime is required"))
        .with_frontend_static_schedule("half_turn_fixture");
        let build = DetectorErrorModel::builder()
            .with_program(program)
            .with_qubits(2)
            .with_noise(NoiseConfig::new(0.0, 0.2, 0.0, 0.0))
            .with_idle_after_2q(0.001)
            .with_detectors(vec![vec![0], vec![1]])
            .with_observables(vec![])
            .build()
            .unwrap();
        let gates: Vec<_> = build
            .prepared()
            .circuit()
            .ticks()
            .iter()
            .flat_map(pecos_quantum::Tick::gate_batches)
            .collect();
        assert_eq!(
            gates
                .iter()
                .filter(|gate| gate.gate_type == GateType::RZZ)
                .count(),
            1
        );
        assert_eq!(
            gates
                .iter()
                .filter(|gate| gate.gate_type == GateType::Idle)
                .map(|gate| gate.qubits.len())
                .sum::<usize>(),
            2
        );
        assert!(build.dem().to_string().contains("error("));
    }
}

#[test]
fn hosted_order_validation_is_opt_in_during_native_preparation() {
    let ir = r#"
        @host_key = private constant [7 x i8] c"host_id"
        @role_key = private constant [10 x i8] c"local_role"
        @host_id = private constant [6 x i8] c"host:a"
        @role = private constant [12 x i8] c"basis_prefix"
        define i64 @qmain(i64 %shot) #0 {
            call void @pecos_qis_trace_metadata_direct(ptr @host_key, i64 7, ptr @host_id, i64 6)
            call void @__quantum__qis__reset__body(i64 0)
            call void @pecos_qis_trace_metadata_direct(ptr @host_key, i64 7, ptr @host_id, i64 6)
            call void @pecos_qis_trace_metadata_direct(ptr @role_key, i64 10, ptr @role, i64 12)
            %m = call i32 @__quantum__qis__m__body(i64 0, i64 0)
            ret i64 0
        }
        declare void @pecos_qis_trace_metadata_direct(ptr, i64, ptr, i64)
        declare void @__quantum__qis__reset__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        attributes #0 = { "EntryPoint" }
    "#;
    for required in [false, true] {
        let result = DetectorErrorModel::builder()
            .with_program(
                program(ir)
                    .with_frontend_static_schedule("test_fixture_preflight")
                    .with_hosted_validation_context("native hosted fixture")
                    .with_schedule_validation_shots(1)
                    .with_require_hosted_operation_order(required),
            )
            .with_qubits(1)
            .with_detectors(vec![vec![0]])
            .with_observables(vec![])
            .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001))
            .build();
        if required {
            let error = result.err().unwrap().to_string();
            assert!(error.contains("ordering drift"));
            assert!(error.contains("native hosted fixture"));
        } else {
            assert!(result.is_ok(), "{}", result.err().unwrap());
        }
    }
}

#[test]
fn prepared_random_readouts_are_rejected_for_both_roles() {
    let ir = r#"
        define i64 @qmain(i64 %shot) #0 {
            call void @__quantum__qis__reset__body(i64 0)
            call void @__quantum__qis__h__body(i64 0)
            %m = call i32 @__quantum__qis__m__body(i64 0, i64 0)
            ret i64 0
        }
        declare void @__quantum__qis__reset__body(i64)
        declare void @__quantum__qis__h__body(i64)
        declare i32 @__quantum__qis__m__body(i64, i64)
        attributes #0 = { "EntryPoint" }
    "#;
    for detector in [false, true] {
        let builder = DetectorErrorModel::builder()
            .with_program(program(ir).with_frontend_static_schedule("test_fixture_preflight"))
            .with_qubits(1)
            .with_noise(NoiseConfig::new(0.001, 0.01, 0.001, 0.001));
        let builder = if detector {
            builder.with_detectors(vec![vec![0]])
        } else {
            builder.with_observables(vec![vec![0]])
        };
        let role = if detector { "detector" } else { "observable" };
        assert_eq!(
            builder.build().err().unwrap().to_string(),
            format!(
                "{role} 0 has a random noiseless parity; detectors and observables must be deterministic in the noiseless circuit"
            )
        );
    }
}

#[test]
#[cfg(feature = "hugr")]
fn generator_certificate_binds_exact_executed_hugr_bytes() {
    use pecos_qec::dem_build::generator_certificate_digest;
    let bytes = include_bytes!("fixtures/dem_programs/explicit_only.hugr");
    let layout = vec![("measurement".into(), 0)];
    let digest = generator_certificate_digest(bytes, &layout);
    let prepared = DetectorErrorModel::builder()
        .with_program(hugr_program(bytes).with_trusted_generator_layout(layout.clone(), &digest))
        .with_qubits(1)
        .with_detector_definitions(vec![
            DemDefinition::references(
                0,
                vec![pecos_qec::MeasurementRef::tag("measurement").unwrap()],
            )
            .unwrap(),
        ])
        .prepare()
        .unwrap();
    assert_eq!(
        prepared.audit().static_schedule_source.as_deref(),
        Some("generator_layout_v2_program_bound")
    );
    assert_eq!(
        prepared.audit().named_result_binding,
        "generator_layout_v2_program_bound"
    );
    assert_eq!(
        prepared.named_result_measurements()["measurement"],
        [Some(0)]
    );
    for (source, expected) in [
        (bytes.as_slice(), "bad digest"),
        (b"foreign bytes".as_slice(), digest.as_str()),
    ] {
        let error = DetectorErrorModel::builder()
            .with_program(
                hugr_program(source).with_trusted_generator_layout(layout.clone(), expected),
            )
            .with_qubits(1)
            .with_detectors(vec![vec![0]])
            .prepare()
            .err()
            .unwrap();
        assert!(
            error.to_string().contains("certificate does not match"),
            "{error}"
        );
    }
}

#[test]
#[cfg(feature = "hugr")]
fn explicit_hugr_named_and_json_results_use_executed_qis_analysis() {
    for json in [false, true] {
        let builder = DetectorErrorModel::builder()
            .with_program(hugr_program(include_bytes!(
                "fixtures/dem_programs/explicit_only.hugr"
            )))
            .with_qubits(1);
        let builder = if json {
            builder.with_detectors_json(r#"[{"id":0,"result_tags":["measurement"]}]"#)
        } else {
            builder.with_detector_definitions(vec![
                DemDefinition::references(
                    0,
                    vec![pecos_qec::MeasurementRef::tag("measurement").unwrap()],
                )
                .unwrap(),
            ])
        };
        let prepared = builder.prepare().unwrap();
        assert_eq!(
            prepared.named_result_measurements()["measurement"],
            [Some(0)]
        );
        assert_eq!(
            prepared.audit().named_result_binding,
            "compiler_direct_scalar_complete"
        );
        assert_eq!(prepared.definitions().detectors[0].measurements, [0]);
    }
}

#[test]
#[cfg(feature = "hugr")]
fn native_hugr_control_flow_rejects_explicit_before_execution() {
    let error = DetectorErrorModel::builder()
        .with_program(hugr_program(include_bytes!(
            "fixtures/dem_programs/changing_schedule.hugr"
        )))
        .with_qubits(2)
        .with_detectors(vec![vec![0]])
        .prepare()
        .err()
        .unwrap();
    assert!(
        error.to_string().contains(
            "branching or looping control flow cannot be certified from one runtime trace"
        ),
        "{error}"
    );
}

#[test]
#[cfg(feature = "hugr")]
fn frontend_compilation_retains_generator_bytes_and_inference_values() {
    let bytes = include_bytes!("fixtures/dem_programs/parities.hugr");
    let program =
        SeleneDemProgram::from_frontend_compiled_hugr(Qis::from_string(PARITIES), bytes.to_vec())
            .with_runtime(runtime())
            .with_schedule_validation_shots(0);
    let prepared = DetectorErrorModel::builder()
        .with_program(program.clone())
        .with_qubits(3)
        .with_inference(DemInferenceSettings::default())
        .prepare()
        .unwrap();
    assert_eq!(
        prepared.inference_settings().unwrap().raw_tag,
        "raw measurements"
    );
    assert_eq!(
        prepared.detector_outputs().unwrap(),
        [DemOutputSelector {
            tag: "DETECTOR".into(),
            element: 0
        }]
    );
    assert_eq!(
        prepared.source_result_traces().len(),
        prepared.result_trace_values().len()
    );
    for (trace, values) in prepared
        .source_result_traces()
        .iter()
        .zip(prepared.result_trace_values())
    {
        assert_eq!(trace.width, values.len());
    }
    let explicit_bytes = include_bytes!("fixtures/dem_programs/explicit_only.hugr");
    let layout = vec![("measurement".to_string(), 0)];
    let digest = pecos_qec::dem_build::generator_certificate_digest(explicit_bytes, &layout);
    let explicit = SeleneDemProgram::from_frontend_compiled_hugr(
        Qis::from_string(include_str!("fixtures/dem_programs/explicit_only.ll")),
        explicit_bytes.to_vec(),
    )
    .with_runtime(runtime())
    .with_trusted_generator_layout(layout, digest);
    let certified = DetectorErrorModel::builder()
        .with_program(explicit)
        .with_qubits(1)
        .with_detector_definitions(vec![
            DemDefinition::references(
                0,
                vec![pecos_qec::MeasurementRef::tag("measurement").unwrap()],
            )
            .unwrap(),
        ])
        .prepare()
        .unwrap();
    assert_eq!(
        certified.audit().named_result_binding,
        "generator_layout_v2_program_bound"
    );
    assert_eq!(
        certified.named_result_measurements()["measurement"],
        [Some(0)]
    );
    let error = DetectorErrorModel::builder()
        .with_program(program.with_trusted_generator_layout(vec![], "bad digest"))
        .with_qubits(3)
        .with_detectors(vec![])
        .prepare()
        .err()
        .unwrap();
    assert!(
        error.to_string().contains("certificate does not match"),
        "{error}"
    );
}

#[test]
fn explicit_qir_slot_preserves_raw_name_and_occurrence_identity() {
    use pecos_engines::{sim_builder, state_vector};
    use pecos_qis::{helios_interface_builder, qis_engine};

    // Follow the QIR case in selene_request_tests.rs: static program qubits,
    // explicit result slots, and __quantum__rt__result_get_one reads.
    let ir = include_str!("fixtures/dem_programs/sparse_qir_slot.ll");
    let prepared = DetectorErrorModel::builder()
        .with_program(program(ir).with_frontend_static_schedule("test_frontend"))
        .with_qubits(1)
        .with_detectors(vec![vec![0]])
        .prepare()
        .unwrap();
    assert_eq!(prepared.measurement_ids(), [0]);
    assert_eq!(
        prepared.circuit().get_meta("qis_source_measurement_ids"),
        Some(&Attribute::String("[0]".into()))
    );
    assert!(prepared.named_result_measurements().is_empty());

    // Slot 7 survives in the engine's raw result name. It is not the DEM
    // occurrence ID and does not create a named-result binding.
    let shots = sim_builder()
        .classical(
            qis_engine()
                .runtime(runtime())
                .interface(helios_interface_builder())
                .program_ir(ir.to_string()),
        )
        .quantum(state_vector())
        .qubits(1)
        .workers(1)
        .seed(42)
        .run(1)
        .unwrap();
    assert_eq!(shots.shots.len(), 1);
    assert_eq!(shots.shots[0].data.len(), 1);
    assert_eq!(shots.shots[0].data["measurement_7"].as_u32(), Some(0));
}
