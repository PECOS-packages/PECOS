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
fn idle_noise_needs_idle_gates() {
    let build = |idle_after_2q: Option<f64>| {
        let builder = DetectorErrorModel::builder()
            .with_circuit(circuit())
            .with_detectors(vec![vec![0, 1]])
            .with_observables(vec![vec![0]])
            .with_noise(NoiseConfig::new(0.001, 0.01, 0.02, 0.001).set_idle(0.01));
        match idle_after_2q {
            Some(duration) => builder.with_idle_after_2q(duration).build(),
            None => builder.build(),
        }
    };
    assert!(build(Some(1.0)).is_ok());
    let Err(error) = build(None) else {
        panic!("idle noise without idle gates must be rejected");
    };
    assert!(
        error
            .to_string()
            .contains("idle-noise parameters have no idle gates to attach to"),
        "{error}"
    );
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
            resolved_detector_outputs: None,
            resolved_observable_outputs: None,
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
            resolved_detector_outputs: None,
            resolved_observable_outputs: None,
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
            resolved_detector_outputs: None,
            resolved_observable_outputs: None,
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

#[test]
fn typed_metadata_survives_explicit_inferred_and_mixed_builds() {
    use pecos_quantum::Attribute;
    use serde_json::{Value, json};

    // Exercise both pure paths and both choices of the inferred role in a mix.
    for (infer_detector, infer_observable) in
        [(false, false), (true, true), (false, true), (true, false)]
    {
        let detector = if infer_detector {
            DemDefinition::inferred(0, "checks", 2)
        } else {
            DemDefinition::explicit(0, vec![0, 1])
        }
        .with_coords([1.0, 2.0, 3.0])
        .with_metadata("label", "check")
        .with_metadata("custom", json!({"round": 7, "tags": ["memory"]}));
        let observable = if infer_observable {
            DemDefinition::inferred(70, "logical", 3)
        } else {
            DemDefinition::explicit(70, vec![0])
        }
        .with_metadata("label", "logical bit")
        .with_metadata("reference", 1)
        .with_metadata("custom", json!([false, null, 3]));
        let noise = NoiseConfig::new(0.001, 0.01, 0.02, 0.001);
        let mut builder = DetectorErrorModel::builder()
            .with_circuit(circuit())
            .with_detector_definitions(vec![detector])
            .with_observable_definitions(vec![observable])
            .with_noise(noise.clone())
            .with_inference(DemInferenceSettings::default().with_validation_rows(2));
        if infer_detector || infer_observable {
            builder = builder.with_inference_data(probes());
        }
        let build = builder.build().unwrap();
        let explicit = DetectorErrorModel::builder()
            .with_circuit(circuit())
            .with_detectors_json(r#"[{"id":0,"meas_ids":[0,1],"coords":[1,2,3]}]"#)
            .with_observables_json(r#"[{"id":70,"meas_ids":[0]}]"#)
            .with_noise(noise)
            .build()
            .unwrap();
        assert_eq!(build.dem.to_string(), explicit.dem.to_string());
        assert_eq!(
            build.evaluate_runtime_record(&[1, 0]).unwrap(),
            explicit.evaluate_runtime_record(&[1, 0]).unwrap()
        );
        assert_eq!(
            build.inference.is_some(),
            infer_detector || infer_observable
        );
        assert_eq!(
            build.definitions.detectors[0].coords,
            Some(vec![1.0, 2.0, 3.0])
        );
        assert_eq!(
            build.definitions.detectors[0].label.as_deref(),
            Some("check")
        );
        assert_eq!(
            build.definitions.observables[0].label.as_deref(),
            Some("logical bit")
        );
        assert_eq!(build.definitions.observables[0].reference, Some(1));
        for (role, inferred, tag, element, custom) in [
            (
                "detectors",
                infer_detector,
                "checks",
                2,
                json!({"round":7,"tags":["memory"]}),
            ),
            (
                "observables",
                infer_observable,
                "logical",
                3,
                json!([false, null, 3]),
            ),
        ] {
            let Some(Attribute::String(text)) = build.circuit.get_meta(role) else {
                panic!("missing {role} metadata");
            };
            let entries: Value = serde_json::from_str(text).unwrap();
            assert_eq!(entries[0]["custom"], custom);
            if inferred {
                assert_eq!(entries[0]["inferred_from_result_tag"], tag);
                assert_eq!(entries[0]["result_element"], element);
            } else {
                assert!(entries[0].get("inferred_from_result_tag").is_none());
                assert!(entries[0].get("result_element").is_none());
            }
        }
    }
}

struct UnexpectedProgram;
impl DemProgram for UnexpectedProgram {
    fn prepare(&mut self, _: &DemProgramRequest<'_>) -> Result<DemBuildInput, DemBuildError> {
        panic!("invalid definition should fail before preparing a program");
    }
}

#[test]
fn invalid_typed_metadata_fails_before_runtime() {
    for inferred in [false, true] {
        let definition = if inferred {
            DemDefinition::inferred(0, "checks", 0)
        } else {
            DemDefinition::explicit(0, vec![0])
        };
        for key in [
            "id",
            "records",
            "meas_ids",
            "result_tags",
            "coords",
            "inferred_from_result_tag",
            "result_element",
        ] {
            for detectors in [false, true] {
                let builder = DetectorErrorModel::builder()
                    .with_program(UnexpectedProgram)
                    .with_qubits(2);
                let definitions = vec![definition.clone().with_metadata(key, 0)];
                let result = if detectors {
                    builder.with_detector_definitions(definitions)
                } else {
                    builder.with_observable_definitions(definitions)
                }
                .build();
                assert!(
                    matches!(result, Err(DemBuildError::Configuration(message)) if message.contains("reserved key")),
                    "{key}"
                );
            }
        }
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let result = DetectorErrorModel::builder()
                .with_program(UnexpectedProgram)
                .with_qubits(2)
                .with_detector_definitions(vec![definition.clone().with_coords([0.0, value, 0.0])])
                .build();
            assert!(
                matches!(result, Err(DemBuildError::Configuration(message)) if message.contains("finite"))
            );
        }
        let result = DetectorErrorModel::builder()
            .with_program(UnexpectedProgram)
            .with_qubits(2)
            .with_observable_definitions(vec![definition.with_coords([0.0, 0.0, 0.0])])
            .build();
        assert!(
            matches!(result, Err(DemBuildError::Configuration(message)) if message.contains("coordinates require a detector"))
        );
    }
}

#[test]
fn every_partition_of_explicit_and_inferred_definitions_matches_the_parity_oracle() {
    use pecos_decoder_core::obs_mask::ObsMask;

    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0, 1, 2, 3]);
    circuit.tick().h(&[0]);
    circuit.tick().cx(&[(0, 1), (2, 3)]);
    circuit.tick().mz(&[0, 1, 2, 3]);
    pass::assign_missing_meas_ids(&mut circuit);
    let detector_supports = [vec![1], vec![0, 2], vec![1, 2, 3], vec![0, 1, 2, 3]];
    let observable_supports = [vec![0], vec![1, 3], vec![0, 2, 3], vec![0, 1, 2, 3]];
    let observable_ids = [0, 1, 65, 70];
    let raw_ids = vec![2, 0, 3, 1];
    let parity = |record: usize, support: &[usize]| -> u8 {
        support
            .iter()
            .fold(0, |value, &id| value ^ u8::from((record >> id) & 1 != 0))
    };
    let noise = NoiseConfig::new(0.001, 0.01, 0.02, 0.001);
    let explicit = DetectorErrorModel::builder()
        .with_circuit(circuit.clone())
        .with_detectors(detector_supports.to_vec())
        .with_observables_json(
            serde_json::json!(
                observable_ids
                    .iter()
                    .zip(&observable_supports)
                    .map(|(id, ids)| serde_json::json!({"id": id, "meas_ids": ids}))
                    .collect::<Vec<_>>()
            )
            .to_string(),
        )
        .with_noise(noise.clone())
        .build()
        .unwrap();
    assert!(explicit.dem.to_string().contains("error("));

    // All 256 source assignments, including both pure cases and role-wise mixing.
    for partition in 0_usize..256 {
        let mut definitions = [Vec::new(), Vec::new()];
        let mut rows = [vec![Vec::new(); 16], vec![Vec::new(); 16]];
        for (role, supports) in [&detector_supports, &observable_supports]
            .iter()
            .enumerate()
        {
            // Reverse definition order to decouple inference columns from output IDs.
            for index in (0..4).rev() {
                let id = if role == 0 {
                    index
                } else {
                    observable_ids[index]
                };
                let definition = if partition & (1 << (4 * role + index)) != 0 {
                    for (record, row) in rows[role].iter_mut().enumerate() {
                        row.push(parity(record, &supports[index]));
                    }
                    DemDefinition::inferred(id, if role == 0 { "checks" } else { "logical" }, index)
                } else {
                    DemDefinition::explicit(id, supports[index].clone())
                };
                definitions[role].push(definition);
            }
        }
        let [detectors, observables] = definitions;
        let [detector_rows, observable_rows] = rows;
        let mut builder = DetectorErrorModel::builder()
            .with_circuit(circuit.clone())
            .with_detector_definitions(detectors)
            .with_observable_definitions(observables)
            .with_noise(noise.clone())
            .with_inference(DemInferenceSettings::default().with_validation_rows(2));
        if partition != 0 {
            builder = builder.with_inference_data(DemInferenceData {
                raw_measurement_ids: raw_ids.clone(),
                raw_rows: (0..16)
                    .map(|record| raw_ids.iter().map(|&id| parity(record, &[id])).collect())
                    .collect(),
                detector_rows,
                observable_rows,
            });
        }
        let build = builder.build().unwrap();
        assert_eq!(build.inference.is_some(), partition != 0);
        assert_eq!(
            build.dem.to_string(),
            explicit.dem.to_string(),
            "partition {partition}"
        );
        for record in 0..16 {
            let bits: Vec<_> = (0..4).map(|id| parity(record, &[id])).collect();
            let expected_detectors: Vec<_> = detector_supports
                .iter()
                .map(|ids| parity(record, ids))
                .collect();
            let mut expected_observables = ObsMask::new();
            for (&id, support) in observable_ids.iter().zip(&observable_supports) {
                if parity(record, support) != 0 {
                    expected_observables.set(id);
                }
            }
            assert_eq!(
                build.evaluate_runtime_record(&bits).unwrap(),
                (expected_detectors, expected_observables),
                "partition {partition}, record {record}"
            );
        }
    }
}

#[test]
fn sparse_detector_ids_rejected_from_json_and_circuit_metadata() {
    let json = r#"[{"id":4,"meas_ids":[0]}]"#;
    for from_circuit in [false, true] {
        let mut input = circuit();
        if from_circuit {
            input.set_meta("detectors", pecos_quantum::Attribute::String(json.into()));
        }
        let builder = DetectorErrorModel::builder().with_circuit(input);
        let builder = if from_circuit {
            builder
        } else {
            builder.with_detectors_json(json)
        };
        let error = builder.build().err().unwrap().to_string();
        assert!(
            error.contains('4') && error.contains("dense from zero"),
            "{error}"
        );
    }
}

#[test]
fn existing_definitions_reject_any_builder_role_or_inference() {
    for kind in 0..4 {
        let mut input = circuit();
        let carried = match kind {
            0 => {
                input.set_meta(
                    "detectors",
                    pecos_quantum::Attribute::String(r#"[{"id":0,"meas_ids":[0]}]"#.into()),
                );
                "detectors metadata"
            }
            1 => {
                input.set_meta(
                    "observables",
                    pecos_quantum::Attribute::String(r#"[{"id":0,"meas_ids":[0]}]"#.into()),
                );
                "observables metadata"
            }
            2 => {
                let reference = input.meas_ref(2, 0, pecos_core::QubitId(0)).unwrap();
                input.detector(&[reference]).unwrap();
                "Detector annotation"
            }
            _ => {
                let reference = input.meas_ref(2, 0, pecos_core::QubitId(0)).unwrap();
                input.observable(&[reference]).unwrap();
                "Observable annotation"
            }
        };
        for role in 0..3 {
            let builder = DetectorErrorModel::builder().with_circuit(input.clone());
            let builder = match role {
                0 => builder.with_detectors(vec![vec![0]]),
                1 => builder.with_observables(vec![vec![0]]),
                _ => builder
                    .with_inference(DemInferenceSettings::default().with_validation_rows(2))
                    .with_inference_data(probes()),
            };
            let error = builder.build().err().unwrap().to_string();
            assert!(error.contains(carried), "{error}");
        }
        assert!(
            DetectorErrorModel::builder()
                .with_circuit(input)
                .build()
                .is_ok()
        );
    }
}

#[test]
fn tracked_pauli_is_legal_with_explicit_and_inferred_definitions() {
    for inferred in [false, true] {
        let mut input = circuit();
        input.tracked_pauli("Z".parse::<pecos_core::PauliString>().unwrap());
        let builder = DetectorErrorModel::builder().with_circuit(input);
        let builder = if inferred {
            builder
                .with_inference(DemInferenceSettings::default().with_validation_rows(2))
                .with_inference_data(probes())
        } else {
            builder.with_detectors(vec![vec![0]])
        };
        let build = builder.build().unwrap();
        assert_eq!(build.circuit.annotations().len(), 1);
    }
}

struct MismatchedSelectors {
    role: &'static str,
    count: usize,
}

impl DemProgram for MismatchedSelectors {
    fn prepare(&mut self, _: &DemProgramRequest<'_>) -> Result<DemBuildInput, DemBuildError> {
        let selectors = vec![
            pecos_qec::DemOutputSelector {
                tag: "test".into(),
                element: 0
            };
            self.count
        ];
        Ok(DemBuildInput {
            circuit: circuit(),
            probes: Some(probes()),
            resolved_detector_outputs: (self.role == "detector").then_some(selectors.clone()),
            resolved_observable_outputs: (self.role == "observable").then_some(selectors),
        })
    }
}

#[test]
fn resolved_selector_counts_must_match_supports() {
    for role in ["detector", "observable"] {
        for count in [0, 2] {
            let error = DetectorErrorModel::builder()
                .with_program(MismatchedSelectors { role, count })
                .with_qubits(2)
                .with_inference(DemInferenceSettings::default().with_validation_rows(2))
                .build()
                .err()
                .unwrap()
                .to_string();
            assert!(
                error.contains(&format!(
                    "resolved {role} selector count {count} differs from inferred support count 1"
                )),
                "{error}"
            );
        }
    }
}

#[test]
fn supplied_probes_retain_default_column_origins() {
    let build = DetectorErrorModel::builder()
        .with_circuit(circuit())
        .with_inference(DemInferenceSettings::default().with_validation_rows(2))
        .with_inference_data(probes())
        .build()
        .unwrap();
    for origin in build.definition_origins {
        assert!(matches!(
            origin.source,
            pecos_qec::DemAnnotationOrigin::InferredDefaults { column: 0, .. }
        ));
    }
}
