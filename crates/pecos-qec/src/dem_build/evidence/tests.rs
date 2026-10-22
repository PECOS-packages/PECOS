// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use super::*;
use crate::dem_build::{DemBuildBuilder, MeasurementRef, ResultRef};
use crate::{
    DemBuildInput, DemDefinition, DemInferenceData, DemProgram, DemProgramRequest,
    DetectorErrorModel,
};
use pecos_core::{Gate, MeasId};
use pecos_quantum::TickCircuit;
use std::collections::BTreeMap;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

struct Program {
    input: Option<DemBuildInput>,
    calls: Arc<AtomicUsize>,
}

impl DemProgram for Program {
    fn static_schedule_evidence(&self) -> Result<Option<String>, DemBuildError> {
        Ok(self
            .input
            .as_ref()
            .unwrap()
            .evidence
            .schedule
            .static_source
            .clone())
    }

    fn prepare(&mut self, _: &DemProgramRequest<'_>) -> Result<DemBuildInput, DemBuildError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(self.input.take().unwrap())
    }
}

fn input(static_source: bool) -> DemBuildInput {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0, 1]);
    let mut measurement = Gate::mz(&[0, 1]);
    measurement.meas_ids = vec![MeasId::from_raw(7), MeasId::from_raw(3)].into();
    circuit.tick().try_add_gate(measurement).unwrap();
    DemBuildInput {
        circuit,
        evidence: DemExecutionEvidence {
            schedule: DemScheduleEvidence {
                static_source: static_source.then(|| "custom_static_analysis".into()),
                validation_shots: 0,
            },
            ..Default::default()
        },
        source_measurement_ids: vec![3, 7],
        result_traces: vec![call("m", 1), call("m", 1)],
        result_tag_analysis: Some(Ok((
            BTreeMap::from([("m".into(), vec![Some(0), Some(1)])]),
            2,
        ))),
        ..Default::default()
    }
}

fn call(name: &str, width: usize) -> ResultTrace {
    ResultTrace {
        name: name.into(),
        occurrence: None,
        width,
        measurement_ids: None,
    }
}

fn inferred_input(static_source: bool) -> DemBuildInput {
    let mut input = input(static_source);
    let rows: Vec<_> = (0..8).map(|row| vec![row & 1, (row >> 1) & 1]).collect();
    input.probes = Some(DemInferenceData {
        raw_measurement_ids: vec![3, 7],
        detector_rows: rows.iter().map(|row| vec![row[0] ^ row[1]]).collect(),
        observable_rows: rows.iter().map(|row| vec![row[0]]).collect(),
        raw_rows: rows,
    });
    input.evidence.raw_binding = Some("custom_bound_ids".into());
    input
        .result_traces
        .extend([call("raw measurements", 1), call("raw measurements", 1)]);
    input
}

fn builder(input: DemBuildInput) -> (DemBuildBuilder, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    (
        DetectorErrorModel::builder()
            .with_program(Program {
                input: Some(input),
                calls: calls.clone(),
            })
            .with_qubits(2),
        calls,
    )
}

fn settings() -> DemInferenceSettings {
    DemInferenceSettings::default().with_validation_rows(2)
}

#[test]
fn explicit_policy_requires_static_before_execution_even_with_settings() {
    for static_source in [false, true] {
        let (builder, calls) = builder(input(static_source));
        let result = builder
            .with_inference(settings())
            .with_detectors(vec![vec![3]])
            .with_observables(vec![])
            .prepare();
        assert_eq!(calls.load(Ordering::SeqCst), usize::from(static_source));
        assert_eq!(result.is_ok(), static_source);
        if let Ok(prepared) = result {
            assert_eq!(
                prepared.audit().static_schedule_source.as_deref(),
                Some("custom_static_analysis")
            );
            assert_eq!(prepared.audit().schedule_validation_shots, None);
        }
    }
}

#[test]
fn mixed_named_policy_requires_static_before_execution() {
    for json in [
        None,
        Some(r#"[{"id":0,"result_tags":["m"]}]"#),
        Some(r#"[{"id":0,"result\u005ftags":["m"]}]"#),
    ] {
        for static_source in [false, true] {
            let (builder, calls) = builder(inferred_input(static_source));
            let builder = builder
                .with_inference(settings())
                .with_detector_definitions(vec![DemDefinition::inferred(0, "check", 0)]);
            let builder = if let Some(json) = json {
                builder.with_observables_json(json)
            } else {
                builder.with_observable_definitions(vec![
                    DemDefinition::references(0, vec![MeasurementRef::tag("m").unwrap()]).unwrap(),
                ])
            };
            let result = builder.prepare();
            assert_eq!(calls.load(Ordering::SeqCst), usize::from(static_source));
            assert_eq!(result.is_ok(), static_source);
            if let Ok(prepared) = result {
                assert_eq!(prepared.audit().schedule_policy, "static_required");
                assert_eq!(
                    prepared.audit().soundness_scope,
                    "empirical_static_schedule"
                );
            }
        }
    }
}

#[test]
fn mixed_positional_policy_is_empirical() {
    let mut input = inferred_input(false);
    input.evidence.schedule.validation_shots = 8;
    input.evidence.provenance_shots = Some(0);
    let (builder, calls) = builder(input);
    let prepared = builder
        .with_inference(settings())
        .with_detector_definitions(vec![
            DemDefinition::inferred(0, "check", 0),
            DemDefinition::explicit(1, vec![7]),
        ])
        .with_observables(vec![])
        .prepare()
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let audit = prepared.audit();
    assert_eq!(audit.soundness_scope, "empirical_static_schedule");
    assert_eq!(audit.schedule_policy, "empirical");
    assert_eq!(audit.schedule_validation_shots, Some(8));
    assert_eq!(audit.named_result_binding, "custom_bound_ids");
    assert_eq!(audit.inference.unwrap().provenance_shots, Some(0));
}

#[test]
fn pure_inference_policy_does_not_invent_schedule_evidence() {
    let (builder, calls) = builder(inferred_input(false));
    let prepared = builder.with_inference(settings()).prepare().unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(prepared.audit().soundness_scope, "unverified_schedule");
    assert_eq!(prepared.audit().schedule_validation_shots, None);
}

#[test]
fn fixed_circuit_probes_are_caller_bound_without_sampled_evidence() {
    let input = inferred_input(false);
    let prepared = DetectorErrorModel::builder()
        .with_circuit(input.circuit)
        .with_inference_data(input.probes.unwrap())
        .with_inference(settings())
        .prepare()
        .unwrap();
    let audit = prepared.audit();
    assert_eq!(
        audit.static_schedule_source.as_deref(),
        Some("fixed_circuit")
    );
    assert_eq!(audit.schedule_validation_shots, None);
    assert_eq!(audit.named_result_binding, "caller_bound_measurement_ids");
    assert_eq!(audit.inference.unwrap().schedule_validation, None);
}

#[test]
fn compiler_calls_holes_and_sparse_permuted_ids_resolve_in_prepare() {
    let mut input = input(true);
    pecos_quantum::pass::normalize_traced_tick_circuit(&mut input.circuit, true);
    input.result_traces.insert(1, call("m", 2));
    input.result_tag_analysis = Some(Ok((
        BTreeMap::from([("m".into(), vec![Some(0), None, Some(1)])]),
        2,
    )));
    let (builder, _) = builder(input);
    let prepared = builder
        .with_detector_definitions(vec![
            DemDefinition::references(
                0,
                vec![MeasurementRef::Result(
                    ResultRef::new("m", 2, None).unwrap(),
                )],
            )
            .unwrap(),
        ])
        .prepare()
        .unwrap();
    assert_eq!(prepared.measurement_ids(), [7, 3]);
    assert_eq!(prepared.definitions().detectors[0].measurements, [0]);
    assert_eq!(
        prepared
            .evaluate_measurements(&BTreeMap::from([(7, 1), (3, 0)]))
            .unwrap()
            .0,
        [1]
    );
    assert_eq!(
        prepared.named_result_measurements()["m"],
        [Some(3), None, None, Some(7)]
    );
    assert_eq!(
        prepared.audit().named_result_binding,
        "compiler_direct_scalar_partial"
    );
    assert_eq!(
        prepared.measurement_ledger()[0].result_refs[0].occurrence(),
        2
    );
}

#[test]
fn inference_preserves_raw_call_boundaries_and_checks_certified_disagreement() {
    for disagree in [false, true] {
        let mut input = inferred_input(true);
        let (tags, _) = input
            .result_tag_analysis
            .as_mut()
            .unwrap()
            .as_mut()
            .unwrap();
        tags.insert(
            "raw measurements".into(),
            if disagree {
                vec![Some(1), Some(0)]
            } else {
                vec![Some(0), Some(1)]
            },
        );
        let (builder, _) = builder(input);
        let result = builder
            .with_inference(settings())
            .with_observable_definitions(vec![
                DemDefinition::references(0, vec![MeasurementRef::tag("m").unwrap()]).unwrap(),
            ])
            .prepare();
        if disagree {
            assert!(
                result
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("disagrees with inference binding")
            );
        } else {
            let prepared = result.unwrap();
            let refs = &prepared.measurement_ledger()[0].result_refs;
            let raw = refs
                .iter()
                .find(|reference| reference.tag() == "raw measurements")
                .unwrap();
            assert_eq!(raw.occurrence(), 1);
            assert_eq!(raw.element(), None);
        }
    }
}

#[test]
fn optional_certification_errors_drop_enrichment_but_required_references_fail() {
    for required in [false, true] {
        let mut input = input(true);
        input.result_tag_analysis = Some(Err(DemBindingError::Certification(
            "unsupported control flow".into(),
        )));
        let (builder, _) = builder(input);
        let builder = if required {
            builder.with_detector_definitions(vec![
                DemDefinition::references(0, vec![MeasurementRef::tag("m").unwrap()]).unwrap(),
            ])
        } else {
            builder.with_detectors(vec![vec![3]])
        };
        let result = builder.prepare();
        if required {
            assert_eq!(
                result.err().unwrap().to_string(),
                "Cannot bind result_ref: the program's lowered form could not be certified; use records, rec[-k], or meas_ids instead."
            );
        } else {
            assert!(result.unwrap().named_result_measurements().is_empty());
        }
    }
    let mut input = input(true);
    input.result_tag_analysis = Some(Err(DemBindingError::Value("bad evidence".into())));
    let (builder, _) = builder(input);
    assert_eq!(
        builder
            .with_detectors(vec![vec![3]])
            .prepare()
            .err()
            .unwrap()
            .to_string(),
        "bad evidence"
    );
}

#[test]
fn generator_arrays_are_certified_as_synthetic_scalar_tags() {
    for foreign in [false, true] {
        let mut input = input(true);
        let layout = vec![
            ("raw measurements:meas:0".into(), 0),
            ("raw measurements:meas:1".into(), 0),
        ];
        input.result_traces = vec![
            call("raw measurements", 2),
            call("raw measurements:meas:0", 1),
            call("raw measurements:meas:1", 1),
        ];
        let expected_digest = generator_certificate_digest(b"exact HUGR bytes", &layout);
        input.generator_layout = Some(GeneratorLayout {
            hugr_bytes: if foreign {
                b"foreign bytes".to_vec()
            } else {
                b"exact HUGR bytes".to_vec()
            },
            layout,
            expected_digest,
        });
        let (builder, _) = builder(input);
        let result = builder
            .with_detector_definitions(vec![
                DemDefinition::references(
                    0,
                    vec![MeasurementRef::tag("raw measurements:meas:1").unwrap()],
                )
                .unwrap(),
            ])
            .prepare();
        if foreign {
            assert!(
                result
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("certificate does not match")
            );
        } else {
            let prepared = result.unwrap();
            assert_eq!(prepared.definitions().detectors[0].measurements, [0]);
            assert_eq!(
                prepared
                    .evaluate_measurements(&BTreeMap::from([(7, 1), (3, 0)]))
                    .unwrap()
                    .0,
                [1]
            );
            assert_eq!(
                prepared.audit().named_result_binding,
                "generator_layout_v2_program_bound"
            );
        }
    }
}

#[test]
fn required_holes_and_json_certification_fail_closed() {
    let mut hole = input(true);
    hole.result_tag_analysis = Some(Ok((BTreeMap::from([("m".into(), vec![None, Some(1)])]), 2)));
    let (builder, _) = builder(hole);
    let error = builder
        .with_detector_definitions(vec![
            DemDefinition::references(0, vec![MeasurementRef::tag("m").unwrap()]).unwrap(),
        ])
        .prepare()
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("not a direct scalar measurement result")
    );
    let mut uncertified = input(true);
    uncertified.result_tag_analysis =
        Some(Err(DemBindingError::Certification("unsupported".into())));
    let (builder, _) = self::builder(uncertified);
    let error = builder
        .with_detectors_json(r#"[{"id":0,"result_tags":["m"]}]"#)
        .prepare()
        .err()
        .unwrap();
    assert_eq!(error.to_string(), JSON_CERTIFICATION_ERROR);
}

#[test]
fn pure_inference_with_static_evidence_keeps_empirical_policy() {
    let (builder, _) = builder(inferred_input(true));
    let prepared = builder.with_inference(settings()).prepare().unwrap();
    let audit = prepared.audit();
    assert_eq!(audit.soundness_scope, "empirical_static_schedule");
    assert_eq!(audit.schedule_policy, "empirical");
    assert_eq!(
        audit.static_schedule_source.as_deref(),
        Some("custom_static_analysis")
    );
    assert_eq!(audit.schedule_validation_shots, None);
}

#[test]
fn custom_adapter_unnumbered_traces_are_cross_checked_without_panicking() {
    let mut input = inferred_input(true);
    input.result_tag_analysis = None;
    for (call, id) in input.result_traces.iter_mut().zip([3, 7, 3, 7]) {
        call.measurement_ids = Some(vec![id]);
    }
    let (builder, _) = builder(input);
    let prepared = builder
        .with_inference(settings())
        .with_observable_definitions(vec![
            DemDefinition::references(0, vec![MeasurementRef::tag("m").unwrap()]).unwrap(),
        ])
        .prepare()
        .unwrap();
    assert_eq!(
        prepared.named_result_measurements()["m"],
        [Some(3), Some(7)]
    );
    assert_eq!(
        prepared.named_result_measurements()["raw measurements"],
        [Some(3), Some(7)]
    );
    let refs = &prepared.measurement_ledger()[0].result_refs;
    assert_eq!(refs.len(), 2);
    assert!(refs.iter().all(|reference| reference.occurrence() == 1));
}

#[test]
fn empirical_routes_skip_compiler_and_generator_tag_certification() {
    for mixed in [false, true] {
        let mut expected_schema = None;
        for analysis in [
            None,
            Some(Ok((
                BTreeMap::from([("m".into(), vec![Some(0), Some(1)])]),
                2,
            ))),
            Some(Err(DemBindingError::Value("invalid analysis".into()))),
            Some(Err(DemBindingError::Certification(
                "unsupported analysis".into(),
            ))),
        ] {
            let mut input = inferred_input(true);
            input.result_tag_analysis = analysis;
            // An empirical request does not consume the adapter's named certificate.
            input.generator_layout = Some(GeneratorLayout {
                hugr_bytes: vec![],
                layout: vec![("m".into(), 0)],
                expected_digest: "invalid digest".into(),
            });
            let (builder, _) = builder(input);
            let builder = builder.with_inference(settings());
            let builder = if mixed {
                builder.with_observables(vec![vec![3]])
            } else {
                builder
            };
            let prepared = builder.prepare().unwrap();
            assert_eq!(
                prepared.named_result_measurements(),
                &BTreeMap::from([("raw measurements".into(), vec![Some(3), Some(7)])])
            );
            assert!(prepared.measurement_ledger().iter().all(|entry| {
                entry.result_refs.len() == 1 && entry.result_refs[0].tag() == "raw measurements"
            }));
            assert_eq!(prepared.audit().named_result_binding, "custom_bound_ids");
            if let Some(expected) = &expected_schema {
                assert_eq!(prepared.schema_fingerprint(), expected);
            } else {
                expected_schema = Some(prepared.schema_fingerprint().to_string());
            }
        }
    }
}

struct ChangingEvidence(Program);

impl DemProgram for ChangingEvidence {
    fn static_schedule_evidence(&self) -> Result<Option<String>, DemBuildError> {
        self.0.static_schedule_evidence()
    }

    fn prepare(&mut self, request: &DemProgramRequest<'_>) -> Result<DemBuildInput, DemBuildError> {
        let mut input = self.0.prepare(request)?;
        input.evidence.schedule.static_source = Some("changed_source".into());
        Ok(input)
    }
}

#[test]
fn adapter_cannot_change_static_evidence_during_preparation() {
    let calls = Arc::new(AtomicUsize::new(0));
    let error = DetectorErrorModel::builder()
        .with_program(ChangingEvidence(Program {
            input: Some(input(true)),
            calls: calls.clone(),
        }))
        .with_qubits(2)
        .with_detectors(vec![vec![3]])
        .prepare()
        .err()
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        error.to_string(),
        "adapter static schedule evidence changed during preparation"
    );
}

#[test]
fn custom_adapter_missing_static_evidence_has_source_neutral_guidance() {
    let (builder, calls) = builder(input(false));
    let error = builder
        .with_detectors(vec![vec![3]])
        .prepare()
        .err()
        .unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        error.to_string(),
        "DEM program requires static schedule evidence; the adapter must report a nonempty static schedule source before execution"
    );
}

#[test]
fn json_result_tags_count_guard_runs_through_preparation() {
    for count in [1, 2, 3] {
        let mut input = input(true);
        // No runtime tags: optional compiler enrichment skips its count check.
        // JSON result_tags must still enforce the independent resolution guard.
        input.result_traces.clear();
        input
            .result_tag_analysis
            .as_mut()
            .unwrap()
            .as_mut()
            .unwrap()
            .1 = count;
        let (builder, calls) = builder(input);
        let result = builder
            .with_detectors_json(r#"[{"id":0,"result_tags":["m"]}]"#)
            .prepare();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        if count == 2 {
            assert!(result.is_ok());
        } else {
            assert_eq!(
                result.err().unwrap().to_string(),
                format!(
                    "DEM builder parse error: Cannot bind result_tags: the certified lowered program has {count} measurement(s) but the traced program emitted 2 measurement(s)."
                )
            );
        }
    }
}
