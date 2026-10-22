// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use super::*;

fn entries(definitions: &[ResolvedDemSpec]) -> Vec<Value> {
    definitions
        .iter()
        .map(|definition| {
            let mut entry = definition.metadata.clone();
            entry.insert("id".into(), json!(definition.id));
            entry.insert("meas_ids".into(), json!(definition.measurement_ids));
            if let Some(coords) = &definition.coords {
                entry.insert("coords".into(), json!(coords));
            }
            Value::Object(entry)
        })
        .collect()
}

fn spec(refs: Vec<MeasurementRef>) -> DemSpec {
    DemSpec {
        id: None,
        source: DemSpecSource::References(refs),
        coords: None,
        metadata: Map::new(),
    }
}
fn rec(offset: usize) -> MeasurementRef {
    MeasurementRef::record(offset).unwrap()
}
fn tag(name: &str) -> MeasurementRef {
    MeasurementRef::tag(name).unwrap()
}
fn result(name: &str, occurrence: usize, element: Option<usize>) -> MeasurementRef {
    MeasurementRef::Result(ResultRef::new(name, occurrence, element).unwrap())
}
fn trace(name: &str, ids: &[usize]) -> ResultTrace {
    ResultTrace {
        name: name.into(),
        occurrence: None,
        width: ids.len().max(1),
        measurement_ids: (!ids.is_empty()).then(|| ids.to_vec()),
    }
}
fn resolve(
    detectors: &[DemSpec],
    observables: &[DemSpec],
    runtime: &[usize],
    traces: &[ResultTrace],
) -> Result<ResolvedDemSpecs> {
    resolve_dem_specs(detectors, observables, runtime, traces, &BTreeMap::new())
}
fn reordered() -> Vec<ResultTrace> {
    vec![trace("a", &[0]), trace("b", &[1]), trace("c", &[2])]
}

#[test]
fn rec_and_result_refs_resolve_identically_after_runtime_reorder() {
    let records = resolve(
        &[spec(vec![rec(3), rec(1)])],
        &[spec(vec![rec(2)])],
        &[2, 0, 1],
        &reordered(),
    )
    .unwrap();
    let results = resolve(
        &[spec(vec![tag("a"), tag("c")])],
        &[spec(vec![tag("b")])],
        &[2, 0, 1],
        &reordered(),
    )
    .unwrap();
    assert_eq!(records, results);
    assert_eq!(
        entries(&records.detectors),
        vec![json!({"id":0,"meas_ids":[0,2]})]
    );
    assert_eq!(
        records
            .ledger
            .iter()
            .map(|entry| entry.meas_id)
            .collect::<Vec<_>>(),
        [2, 0, 1]
    );
    assert_eq!(
        records.ledger[0].to_json(),
        json!({"meas_id":2,"runtime_record_index":0,"canonical_record_index":2,"result_refs":[{"tag":"c","occurrence":0,"element":null}]})
    );
}

#[test]
fn real_guppy_rec_and_result_ref_builds_are_byte_identical() {
    let traces = [trace("a", &[0]), trace("b", &[1])];
    let records = resolve(
        &[spec(vec![rec(2)])],
        &[spec(vec![rec(1)])],
        &[1, 0],
        &traces,
    )
    .unwrap();
    let named = resolve(
        &[spec(vec![result("a", 0, None)])],
        &[spec(vec![result("b", 0, None)])],
        &[1, 0],
        &traces,
    )
    .unwrap();
    let bare = resolve(
        &[spec(vec![tag("a")])],
        &[spec(vec![tag("b")])],
        &[1, 0],
        &traces,
    )
    .unwrap();
    assert_eq!(records, named);
    assert_eq!(bare, named);
}

#[test]
fn tag_strings_mix_with_rec_and_result_ref_refs() {
    let definition = spec(vec![tag("a"), result("b", 0, None), rec(1)]);
    let resolved = resolve(&[definition], &[], &[2, 0, 1], &reordered()).unwrap();
    assert_eq!(
        entries(&resolved.detectors),
        vec![json!({"id":0,"meas_ids":[0,1,2]})]
    );
}

#[test]
fn reference_constructor_value_errors() {
    assert_eq!(
        MeasurementRef::record(0).unwrap_err().to_string(),
        "measurement record back-offset must be at least one"
    );
    assert_eq!(MeasurementRef::record(1).unwrap(), rec(1));
    assert_eq!(
        MeasurementRef::tag("").unwrap_err().to_string(),
        "result_ref tag must be a non-empty string"
    );
    assert_eq!(
        ResultRef::new("", 0, Some(0)).unwrap_err().to_string(),
        "result_ref tag must be a non-empty string"
    );
    assert_eq!(
        DemSpecSource::references(vec![]).unwrap_err().to_string(),
        "detectors and observables must reference at least one measurement"
    );
    assert_eq!(
        DemSpecSource::inferred("", 0).unwrap_err().to_string(),
        "inferred output tag must be a non-empty string"
    );
}

#[test]
fn maximum_usize_values_preserve_identities_and_check_record_bounds() {
    let reference = result("a", usize::MAX, None);
    let traces = [ResultTrace {
        occurrence: Some(usize::MAX),
        ..trace("a", &[usize::MAX])
    }];
    let observable = DemSpec {
        id: Some(usize::MAX),
        ..spec(vec![reference])
    };
    let schema = resolve(&[], &[observable], &[usize::MAX], &traces).unwrap();
    assert_eq!(schema.observables[0].id, usize::MAX);
    assert_eq!(schema.observables[0].measurement_ids, [usize::MAX]);
    assert_eq!(schema.ledger[0].meas_id, usize::MAX);
    assert_eq!(schema.ledger[0].result_refs[0].occurrence(), usize::MAX);
    assert_eq!(
        resolve(&[spec(vec![rec(usize::MAX)])], &[], &[0], &[])
            .unwrap_err()
            .to_string(),
        format!(
            "measurement record offset -{} is out of range for 1 measurements",
            usize::MAX
        )
    );
    assert_eq!(
        resolve(
            &[spec(vec![result("a", usize::MAX, Some(usize::MAX))])],
            &[],
            &[usize::MAX],
            &traces
        )
        .unwrap_err()
        .to_string(),
        "scalar result_ref \"a\" only accepts element=None or element=0"
    );
    let inferred = DemSpec {
        source: DemSpecSource::inferred("a", usize::MAX).unwrap(),
        ..spec(vec![])
    };
    let supports = BTreeMap::from([(("a".into(), usize::MAX), vec![usize::MAX])]);
    let schema = resolve_dem_specs(&[inferred], &[], &[usize::MAX], &[], &supports).unwrap();
    assert_eq!(schema.detectors[0].measurement_ids, [usize::MAX]);
}

#[test]
fn spec_rejects_dead_parity_and_keeps_first_appearance_order() {
    assert_eq!(
        resolve(&[spec(vec![rec(1), rec(1)])], &[], &[2, 0, 1], &reordered())
            .unwrap_err()
            .to_string(),
        "detector 0 references every measurement with even multiplicity; its parity is identically zero"
    );
    let definition = spec(vec![rec(1), rec(2), rec(1), rec(3), rec(1)]);
    assert_eq!(
        entries(
            &resolve(&[definition], &[], &[2, 0, 1], &[])
                .unwrap()
                .detectors
        ),
        vec![json!({"id":0,"meas_ids":[2,1,0]})]
    );
    // NaN is rejected by serde_json's value construction boundary, not metadata validation.
    assert!(serde_json::Number::from_f64(f64::NAN).is_none());
    assert!(serde_json::Number::from_f64(f64::INFINITY).is_none());
}

#[test]
fn result_ref_rejects_ambiguous_or_missing_runtime_provenance() {
    assert_eq!(
        resolve(
            &[spec(vec![tag("computed")])],
            &[],
            &[2, 0, 1],
            &[trace("computed", &[])]
        )
        .unwrap_err()
        .to_string(),
        "result_ref \"computed\" occurrence 0 is not a direct scalar measurement result"
    );
    assert_eq!(
        resolve(&[spec(vec![tag("missing")])], &[], &[2, 0, 1], &[])
            .unwrap_err()
            .to_string(),
        "result_ref \"missing\" occurrence 0 is absent from the runtime trace"
    );
    assert_eq!(
        resolve(
            &[spec(vec![result("a", 2, None)])],
            &[],
            &[2, 0, 1],
            &reordered()
        )
        .unwrap_err()
        .to_string(),
        "result_ref \"a\" occurrence 2 is absent from the runtime trace"
    );
}

#[test]
fn result_ref_preserves_unsupported_occurrence_holes() {
    let traces = [trace("same", &[]), trace("same", &[0])];
    assert_eq!(
        resolve(&[spec(vec![result("same", 0, None)])], &[], &[0], &traces)
            .unwrap_err()
            .to_string(),
        "result_ref \"same\" occurrence 0 is not a direct scalar measurement result"
    );
    let schema = resolve(&[spec(vec![result("same", 1, None)])], &[], &[0], &traces).unwrap();
    assert_eq!(
        entries(&schema.detectors),
        vec![json!({"id":0,"meas_ids":[0]})]
    );
    assert_eq!(schema.result_ids_by_tag["same"], [None, Some(0)]);
}

#[test]
fn result_ref_rejects_array_element_provenance() {
    for element in [None, Some(0), Some(4)] {
        assert_eq!(
            resolve(
                &[spec(vec![result("pair", 0, element)])],
                &[],
                &[2, 0, 1],
                &[trace("pair", &[0, 1])]
            )
            .unwrap_err()
            .to_string(),
            "array-valued result_ref provenance is not yet certified; expose scalar result() tags or use canonical rec[...] references"
        );
    }
}

#[test]
fn scalar_elements_and_record_bounds() {
    for element in [None, Some(0)] {
        assert!(
            resolve(
                &[spec(vec![result("a", 0, element)])],
                &[],
                &[2, 0, 1],
                &reordered()
            )
            .is_ok()
        );
    }
    assert_eq!(
        resolve(
            &[spec(vec![result("a", 0, Some(1))])],
            &[],
            &[2, 0, 1],
            &reordered()
        )
        .unwrap_err()
        .to_string(),
        "scalar result_ref \"a\" only accepts element=None or element=0"
    );
    assert_eq!(
        resolve(&[spec(vec![rec(4)])], &[], &[2, 0, 1], &[])
            .unwrap_err()
            .to_string(),
        "measurement record offset -4 is out of range for 3 measurements"
    );
}

#[test]
fn scalar_result_provenance_takes_precedence_over_aggregate_arrays() {
    let traces = [
        trace("a", &[0]),
        trace("b", &[1]),
        trace("aggregate", &[1, 0]),
        trace("final", &[2]),
    ];
    let schema = resolve(&[spec(vec![rec(3), rec(1)])], &[], &[2, 0, 1], &traces).unwrap();
    assert_eq!(
        schema.result_ids_by_tag,
        BTreeMap::from([
            ("a".into(), vec![Some(0)]),
            ("b".into(), vec![Some(1)]),
            ("final".into(), vec![Some(2)])
        ])
    );
    assert_eq!(
        schema.ledger[1].result_refs,
        [
            ResultRef::new("a", 0, None).unwrap(),
            ResultRef::new("aggregate", 0, Some(1)).unwrap()
        ]
    );
}

#[test]
fn rec_rejects_non_dense_runtime_measurement_identities() {
    assert_eq!(
        resolve(&[spec(vec![rec(1)])], &[], &[4, 9], &[])
            .unwrap_err()
            .to_string(),
        "rec[...] requires runtime MeasIds to preserve the canonical dense Guppy measurement identity range; use result_ref(...) for non-dense runtimes"
    );
}

#[test]
fn result_ref_supports_non_dense_runtime_measurement_identities() {
    let schema = resolve(
        &[spec(vec![tag("second")])],
        &[],
        &[4, 9],
        &[trace("first", &[4]), trace("second", &[9])],
    )
    .unwrap();
    assert_eq!(
        entries(&schema.detectors),
        vec![json!({"id":0,"meas_ids":[9]})]
    );
    assert!(
        schema
            .ledger
            .iter()
            .all(|entry| entry.canonical_record_index.is_none())
    );
}

#[test]
fn definition_ids_and_error_precedence() {
    let mut definition = spec(vec![rec(1)]);
    definition.id = Some(1);
    assert_eq!(
        resolve(std::slice::from_ref(&definition), &[], &[0], &[])
            .unwrap_err()
            .to_string(),
        "detector ids must be unique and dense from 0 to len(detectors)-1"
    );
    definition.id = Some(0);
    assert_eq!(
        resolve(&[definition.clone(), definition.clone()], &[], &[0], &[])
            .unwrap_err()
            .to_string(),
        "detector ids must be unique and dense from 0 to len(detectors)-1"
    );
    assert_eq!(
        resolve(&[], &[definition.clone(), definition], &[0], &[])
            .unwrap_err()
            .to_string(),
        "observable ids must be unique"
    );
    let mut gap = spec(vec![rec(1)]);
    gap.id = Some(1);
    assert_eq!(
        resolve(&[gap], &[spec(vec![tag("missing")])], &[0], &[])
            .unwrap_err()
            .to_string(),
        "detector ids must be unique and dense from 0 to len(detectors)-1"
    );
}

#[test]
fn reserved_metadata_rejected_for_every_role() {
    for key in [
        "id",
        "records",
        "meas_ids",
        "result_tags",
        "coords",
        "inferred_from_result_tag",
        "result_element",
    ] {
        for role in ["detector", "observable", "inferred_detector"] {
            let mut definition = spec(vec![rec(1)]);
            if role == "inferred_detector" {
                definition.source = DemSpecSource::inferred("checks", 0).unwrap();
            }
            definition.metadata.insert(key.into(), json!(1));
            let supports = BTreeMap::from([(("checks".into(), 0), vec![0])]);
            let (detectors, observables) = if role == "observable" {
                (vec![], vec![definition])
            } else {
                (vec![definition], vec![])
            };
            assert_eq!(
                resolve_dem_specs(&detectors, &observables, &[0], &[], &supports)
                    .unwrap_err()
                    .to_string(),
                format!("detector/observable metadata uses reserved keys: [{key:?}]")
            );
        }
    }
    let metadata = Map::from_iter([("id".into(), json!(1)), ("coords".into(), json!([]))]);
    assert_eq!(
        validate_metadata(&metadata).unwrap_err().to_string(),
        "detector/observable metadata uses reserved keys: [\"coords\", \"id\"]"
    );
}

#[test]
fn inferred_support_required_known_and_nonzero() {
    let definition = DemSpec {
        source: DemSpecSource::inferred("checks", 0).unwrap(),
        ..spec(vec![])
    };
    assert_eq!(
        resolve(std::slice::from_ref(&definition), &[], &[4, 9], &[])
            .unwrap_err()
            .to_string(),
        "detector 0 requires inferred output tag \"checks\" element 0"
    );
    for (ids, message) in [
        (
            vec![0],
            "detector 0 contains unknown inferred measurement IDs",
        ),
        (
            vec![4, 4],
            "detector 0 references every measurement with even multiplicity; its parity is identically zero",
        ),
    ] {
        assert_eq!(
            resolve_dem_specs(
                std::slice::from_ref(&definition),
                &[],
                &[4, 9],
                &[],
                &BTreeMap::from([(("checks".into(), 0), ids)])
            )
            .unwrap_err()
            .to_string(),
            message
        );
    }
}

#[test]
fn mixed_definitions_select_only_requested_elements() {
    let mut inferred = spec(vec![]);
    inferred.source = DemSpecSource::inferred("parities", 0).unwrap();
    let mut logical = spec(vec![]);
    logical.source = DemSpecSource::inferred("logical", 0).unwrap();
    logical.id = Some(70);
    let mut explicit = spec(vec![rec(3)]);
    explicit.id = Some(2);
    let supports = BTreeMap::from([
        (("parities".into(), 0), vec![0, 2]),
        (("logical".into(), 0), vec![1]),
    ]);
    let resolved = resolve_dem_specs(
        &[spec(vec![rec(2)]), inferred],
        &[logical, explicit],
        &[0, 1, 2],
        &[],
        &supports,
    )
    .unwrap();
    assert_eq!(
        resolved
            .definition_origins
            .iter()
            .map(|o| (o.role, o.id))
            .collect::<Vec<_>>(),
        [
            ("detector", 0),
            ("detector", 1),
            ("observable", 2),
            ("observable", 70)
        ]
    );
    assert_eq!(
        resolved
            .definition_origins
            .iter()
            .map(|o| matches!(o.source, DemAnnotationOrigin::Inferred(_)))
            .collect::<Vec<_>>(),
        [false, true, false, true]
    );
    assert_eq!(
        resolved.observables[1].metadata["inferred_from_result_tag"],
        "logical"
    );
}

#[test]
fn arbitrary_coordinates_and_sorted_definitions() {
    for coords in [vec![], vec![1.], vec![1., 2., 3., 4.]] {
        let mut first = spec(vec![rec(1)]);
        first.id = Some(1);
        first.coords = Some(coords.clone());
        let mut second = spec(vec![rec(2)]);
        second.id = Some(0);
        let resolved = resolve(&[first, second], &[], &[0, 1], &[]).unwrap();
        assert_eq!(resolved.detectors[0].id, 0);
        assert_eq!(resolved.detectors[1].coords, Some(coords));
    }
}

#[test]
fn nonfinite_coordinates_are_not_silently_changed_to_null() {
    let mut definition = spec(vec![rec(1)]);
    definition.coords = Some(vec![f64::NAN, f64::INFINITY]);
    let resolved = resolve(&[definition], &[], &[0], &[]).unwrap();
    let coords = resolved.detectors[0].coords.as_ref().unwrap();
    assert!(coords[0].is_nan());
    assert_eq!(coords[1], f64::INFINITY);
}

#[test]
fn mixed_named_explicit_binding_keeps_compiler_provenance() {
    let mut detector = spec(vec![]);
    detector.source = DemSpecSource::inferred("parities", 0).unwrap();
    let supports = BTreeMap::from([(("parities".into(), 0), vec![0, 2])]);
    let traces = [
        trace("raw measurements", &[2, 0, 1]),
        trace("m1", &[1]),
        ResultTrace {
            width: 2,
            ..trace("parities", &[])
        },
    ];
    let schema = resolve_dem_specs(
        &[detector],
        &[spec(vec![tag("m1")])],
        &[0, 1, 2],
        &traces,
        &supports,
    )
    .unwrap();
    assert_eq!(schema.detectors[0].measurement_ids, [0, 2]);
    assert_eq!(schema.observables[0].measurement_ids, [1]);
    assert_eq!(
        resolve(&[spec(vec![tag("parities")])], &[], &[0, 1, 2], &traces)
            .unwrap_err()
            .to_string(),
        "result_ref \"parities\" occurrence 0 is not a direct scalar measurement result"
    );
}

#[test]
fn invalid_selectors_and_duplicate_ids_fail_before_execution() {
    assert_eq!(
        DemSpecSource::inferred("", 0).unwrap_err().to_string(),
        "inferred output tag must be a non-empty string"
    );
    let mut inferred = spec(vec![]);
    inferred.id = Some(0);
    inferred.source = DemSpecSource::inferred("parities", 0).unwrap();
    let mut explicit = spec(vec![rec(1)]);
    explicit.id = Some(0);
    assert_eq!(
        resolve_dem_specs(
            &[inferred, explicit],
            &[],
            &[0, 1, 2],
            &[],
            &BTreeMap::from([(("parities".into(), 0), vec![0, 2])])
        )
        .unwrap_err()
        .to_string(),
        "detector ids must be unique and dense from 0 to len(detectors)-1"
    );
}

#[test]
fn builder_matches_both_wrappers_with_result_tags() {
    let traces = [trace("m0", &[0]), trace("m1", &[1])];
    let schema = resolve(
        &[spec(vec![tag("m0")])],
        &[spec(vec![tag("m1")])],
        &[0, 1],
        &traces,
    )
    .unwrap();
    assert_eq!(schema.detectors[0].measurement_ids, [0]);
    assert_eq!(schema.observables[0].measurement_ids, [1]);
}

#[test]
fn role_mixing_does_not_require_unused_tags() {
    let supports = BTreeMap::from([
        (("parities".into(), 0), vec![0, 2]),
        (("logical".into(), 0), vec![1]),
    ]);
    let mut logical = spec(vec![]);
    logical.source = DemSpecSource::inferred("logical", 0).unwrap();
    let first = resolve_dem_specs(
        &[spec(vec![rec(3), rec(1)])],
        &[logical],
        &[0, 1, 2],
        &[],
        &supports,
    )
    .unwrap();
    assert_eq!(first.detectors[0].measurement_ids, [0, 2]);
    assert_eq!(first.observables[0].measurement_ids, [1]);
    let mut parity = spec(vec![]);
    parity.source = DemSpecSource::inferred("parities", 0).unwrap();
    assert_eq!(
        resolve_dem_specs(&[parity], &[], &[0, 1, 2], &[], &supports)
            .unwrap()
            .observables,
        Vec::<ResolvedDemSpec>::new()
    );
}

#[test]
fn dem_definition_accepts_arbitrary_length_coordinates() {
    for coords in [vec![], vec![1.0], vec![1.0, 2.0, 3.0, 4.0]] {
        let definition =
            crate::dem_build::DemDefinition::explicit(0, vec![0]).with_coords(coords.clone());
        assert_eq!(definition.coords, Some(coords));
    }
}

#[test]
fn mixed_build_does_not_bind_nonraw_arrays() {
    let traces = [trace("raw measurements", &[0, 1]), trace("other", &[0, 1])];
    let mut inferred = spec(vec![]);
    inferred.source = DemSpecSource::inferred("checks", 0).unwrap();
    let schema = resolve_dem_specs(
        &[spec(vec![rec(1)]), inferred],
        &[],
        &[0, 1],
        &traces,
        &BTreeMap::from([(("checks".into(), 0), vec![0])]),
    )
    .unwrap();
    assert_eq!(schema.detectors[0].measurement_ids, [1]);
    assert_eq!(schema.detectors[1].measurement_ids, [0]);
    assert_eq!(
        ResultTraceIndex::new(&[0, 1], &traces)
            .unwrap()
            .named_bindings(Some("raw measurements")),
        BTreeMap::from([("raw measurements".into(), vec![Some(0), Some(1)])])
    );
}
