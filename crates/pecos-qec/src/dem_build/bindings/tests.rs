// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use super::*;

fn trace(
    name: &str,
    occurrence: Option<usize>,
    width: usize,
    ids: Option<&[usize]>,
) -> ResultTrace {
    ResultTrace {
        name: name.into(),
        occurrence,
        width,
        measurement_ids: ids.map(<[usize]>::to_vec),
    }
}
fn scalar(name: &str, occurrence: usize, id: usize) -> ResultTrace {
    trace(name, Some(occurrence), 1, Some(&[id]))
}
fn layout(entries: &[(&str, usize)]) -> Vec<(String, usize)> {
    entries
        .iter()
        .map(|(name, slot)| ((*name).into(), *slot))
        .collect()
}
fn analysis() -> ResultTagMeasurements {
    (
        BTreeMap::from([("a".into(), vec![Some(0)]), ("b".into(), vec![Some(1)])]),
        2,
    )
}

#[test]
fn occurrence_indexing_advances_max_and_preserves_reverse_order() {
    let traces = [
        scalar("a", 2, 4),
        scalar("a", 0, 9),
        trace("a", None, 1, Some(&[4])),
        trace("b", None, 2, Some(&[9, 4])),
    ];
    let indexed = ResultTraceIndex::new(&[9, 4], &traces).unwrap();
    assert_eq!(
        indexed.calls["a"].keys().copied().collect::<Vec<_>>(),
        [0, 2, 3]
    );
    assert!(!indexed.named_bindings(None).contains_key("a"));
    assert_eq!(
        indexed.ledger()[1].result_refs,
        [
            ResultRef::new("a", 2, None).unwrap(),
            ResultRef::new("a", 3, None).unwrap(),
            ResultRef::new("b", 0, Some(1)).unwrap()
        ]
    );
    assert_eq!(
        indexed.ledger()[0].to_json(),
        json!({"meas_id":9,"runtime_record_index":0,"canonical_record_index":null,"result_refs":[{"tag":"a","occurrence":0,"element":null},{"tag":"b","occurrence":0,"element":0}]})
    );
}

#[test]
fn maximum_occurrence_is_explicit_but_exhausts_automatic_occurrences() {
    let traces = [scalar("a", usize::MAX, 4), scalar("a", 0, 9)];
    let index = ResultTraceIndex::new(&[4, 9], &traces).unwrap();
    assert_eq!(index.calls["a"][&usize::MAX], [Some(4)]);
    assert_eq!(index.calls["a"][&0], [Some(9)]);
    let mut exhausted = traces.to_vec();
    exhausted.push(trace("a", None, 1, Some(&[4])));
    assert_eq!(
        ResultTraceIndex::new(&[4, 9], &exhausted)
            .unwrap_err()
            .to_string(),
        "result trace \"a\" has no available occurrence after usize::MAX"
    );
    assert_eq!(
        ResultTraceIndex::new(
            &[4],
            &[scalar("a", usize::MAX, 4), scalar("a", usize::MAX, 4)]
        )
        .unwrap_err()
        .to_string(),
        format!("result trace \"a\" repeats occurrence {}", usize::MAX)
    );
}

#[test]
fn duplicate_occurrences_and_runtime_ids() {
    assert_eq!(
        ResultTraceIndex::new(&[0], &[scalar("a", 0, 0), scalar("a", 0, 0)])
            .unwrap_err()
            .to_string(),
        "result trace \"a\" repeats occurrence 0"
    );
    assert_eq!(
        ResultTraceIndex::new(&[9, 4, 9, 4], &[scalar("a", 0, 0), scalar("a", 0, 0)])
            .unwrap_err()
            .to_string(),
        "traced circuit contains duplicate MeasId(s): [4, 9]"
    );
    let ids: Vec<_> = (0..12).rev().chain(0..12).collect();
    assert_eq!(
        ResultTraceIndex::new(&ids, &[]).unwrap_err().to_string(),
        "traced circuit contains duplicate MeasId(s): [0, 1, 2, 3, 4, 5, 6, 7]"
    );
}

#[test]
fn only_raw_tag_accepts_array_bindings() {
    let traces = [
        trace("raw", Some(0), 2, Some(&[0, 1])),
        trace("other", Some(0), 2, Some(&[0, 1])),
    ];
    let index = ResultTraceIndex::new(&[0, 1], &traces).unwrap();
    assert!(index.named_bindings(None).is_empty());
    assert_eq!(
        index.named_bindings(Some("raw")),
        BTreeMap::from([("raw".into(), vec![Some(0), Some(1)])])
    );
}

#[test]
fn unsupported_widths_dense_occurrences_and_classification() {
    let traces = [
        trace("z", None, 0, None),
        trace("z", None, 3, None),
        scalar("z", 2, 0),
        scalar("sparse", 1, 0),
        trace("empty", None, 0, Some(&[])),
    ];
    let bindings = ResultTraceIndex::new(&[0], &traces)
        .unwrap()
        .named_bindings(None);
    assert_eq!(bindings["z"], [None, None, None, None, Some(0)]);
    assert_eq!(bindings["empty"], [None]);
    assert!(!bindings.contains_key("sparse"));
    assert_eq!(
        named_binding_classification(None, false, &bindings),
        "compiler_direct_scalar_partial"
    );
    assert_eq!(
        named_binding_classification(None, true, &bindings),
        "generator_layout_v2_program_bound"
    );
    assert_eq!(
        named_binding_classification(Some("runtime_result_ids"), true, &bindings),
        "runtime_result_ids"
    );
    for bindings in [
        BTreeMap::new(),
        BTreeMap::from([("a".into(), vec![])]),
        BTreeMap::from([("a".into(), vec![None])]),
    ] {
        assert_eq!(named_binding_classification(None, false, &bindings), "none");
    }
    assert_eq!(
        named_binding_classification(None, false, &BTreeMap::from([("a".into(), vec![Some(0)])])),
        "compiler_direct_scalar_complete"
    );
}

#[test]
fn build_without_candidate_tags_skips_extraction() {
    let result =
        compiler_certified_traces(&[0], &[9], &[], &[], || panic!("analysis must not run"))
            .unwrap();
    assert_eq!(result, Vec::<ResultTrace>::new());
}

#[test]
fn qis_certification_failure_policy_and_unrelated_errors() {
    let traces = [scalar("a", 0, 0), scalar("b", 0, 1)];
    for required in [vec![], vec!["a".into()]] {
        let result = compiler_certified_traces(&[0, 1], &[1, 0], &traces, &required, || {
            Err(DemBindingError::Certification(
                "test unsupported CFG".into(),
            ))
        });
        if required.is_empty() {
            assert_eq!(result.unwrap(), Vec::<ResultTrace>::new());
        } else {
            let error = result.unwrap_err();
            assert_eq!(
                error.to_string(),
                "Cannot bind result_ref: the program's lowered form could not be certified; use records, rec[-k], or meas_ids instead. test unsupported CFG"
            );
            assert!(
                matches!(error,DemBindingError::RequiredCertification { source } if *source == DemBindingError::Certification("test unsupported CFG".into()))
            );
        }
        assert_eq!(
            compiler_certified_traces(&[0, 1], &[1, 0], &traces, &required, || Err(
                DemBindingError::value("unrelated")
            ))
            .unwrap_err()
            .to_string(),
            "unrelated"
        );
    }
}

#[test]
fn certified_measurement_count_mismatch() {
    let traces = [scalar("a", 0, 0), scalar("b", 0, 1)];
    let result = || {
        let (tags, _) = analysis();
        Ok((tags, 3))
    };
    assert_eq!(
        compiler_certified_traces(&[0, 1], &[1, 0], &traces, &[], result).unwrap(),
        Vec::<ResultTrace>::new()
    );
    let error =
        compiler_certified_traces(&[0, 1], &[1, 0], &traces, &["a".into()], result).unwrap_err();
    let DemBindingError::RequiredCertification { source } = error else {
        panic!("expected certification cause");
    };
    assert_eq!(
        source.to_string(),
        "Cannot certify measurement order from the lowered program: the certified lowered program has 3 measurement(s) but the traced program emitted 2 measurement(s)"
    );
    assert_eq!(
        compiler_certified_traces(&[0, 1], &[4, 9], &traces, &[], result)
            .unwrap_err()
            .to_string(),
        "source and runtime measurement identities must be unique and describe the same set"
    );
}

#[test]
fn compiler_encounter_widths_candidates_ordinals_and_absence() {
    let traces = [
        trace("z", Some(88), 0, None),
        trace("z", Some(0), 3, None),
        scalar("ignored", 0, 4),
    ];
    let tags = BTreeMap::from([
        ("z".into(), vec![None, None, Some(1)]),
        ("a".into(), vec![Some(0)]),
    ]);
    let certified =
        compiler_certified_traces(&[4, 9], &[9, 4], &traces, &["a".into()], || Ok((tags, 2)))
            .unwrap();
    assert_eq!(
        certified,
        [
            scalar("a", 0, 4),
            trace("z", Some(0), 1, None),
            trace("z", Some(1), 3, None),
            scalar("z", 2, 9)
        ]
    );
    assert_eq!(
        compiler_certified_traces(
            &[0, 1],
            &[1, 0],
            &[],
            &["missing".into()],
            || Ok(analysis())
        )
        .unwrap_err()
        .to_string(),
        "result_ref \"missing\" is absent from the compiled Guppy program"
    );
    assert_eq!(
        compiler_certified_traces(&[0, 1], &[1, 0], &[], &["a".into()], || Ok((
            BTreeMap::from([("a".into(), vec![Some(2)])]),
            2
        )))
        .unwrap_err()
        .to_string(),
        "compiler measurement ordinal 2 is outside the traced source measurement stream"
    );
}

#[test]
fn integer_prints_preserve_measurement_bindings_and_diagnostics() {
    let tags = BTreeMap::from([
        ("m".into(), vec![Some(0)]),
        ("n".into(), vec![None, None]),
        ("k".into(), vec![Some(1)]),
    ]);
    let runtime = [
        scalar("m", 0, 0),
        trace("n", None, 1, None),
        scalar("k", 0, 1),
    ];
    for required in [vec![], vec!["m".into()]] {
        let traces = compiler_certified_traces(&[0, 1], &[0, 1], &runtime, &required, || {
            Ok((tags.clone(), 2))
        })
        .unwrap();
        let index = ResultTraceIndex::new(&[0, 1], &traces).unwrap();
        let bindings = index.named_bindings(None);
        assert_eq!(bindings["n"], [None, None]);
        assert_eq!(
            named_binding_classification(None, false, &bindings),
            "compiler_direct_scalar_partial"
        );
        assert_eq!(
            index
                .resolve_ref(&MeasurementRef::tag("m").unwrap())
                .unwrap(),
            0
        );
        assert_eq!(
            index
                .resolve_ref(&MeasurementRef::record(1).unwrap())
                .unwrap(),
            1
        );
        assert_eq!(
            index
                .resolve_ref(&MeasurementRef::tag("n").unwrap())
                .unwrap_err()
                .to_string(),
            "result_ref \"n\" occurrence 0 is not a direct scalar measurement result"
        );
    }
}

#[cfg(feature = "selene")]
#[test]
fn parser_failure_without_required_tags_skips_certification() {
    for ir in [
        "define void @qmain(",
        "define void @qmain() {\n%f =\nret void\n}",
    ] {
        let traces = compiler_certified_traces(&[0, 1], &[1, 0], &[scalar("a", 0, 0)], &[], || {
            pecos_qis::text::extract_result_tag_measurements(ir).map_err(Into::into)
        })
        .unwrap();
        assert_eq!(traces, Vec::<ResultTrace>::new());
    }
}

#[test]
fn generator_layout_rejects_source_runtime_identity_mismatch() {
    let entries = layout(&[("first", 0), ("second", 0)]);
    let traces = [scalar("first", 0, 4), scalar("second", 0, 9)];
    assert_eq!(
        generator_certified_traces(&entries, &[0, 1], &[4, 9], &traces, &["first".into()])
            .unwrap_err()
            .to_string(),
        "source and runtime measurement identities must be unique and describe the same set"
    );
    for (source, runtime) in [(vec![0, 0], vec![0, 1]), (vec![0, 1], vec![0, 0])] {
        assert_eq!(
            validate_source_ids(&source, &runtime)
                .unwrap_err()
                .to_string(),
            "source and runtime measurement identities must be unique and describe the same set"
        );
    }
}

#[test]
fn generator_layout_rejects_permuted_array_runtime_result_ids() {
    assert_eq!(
        generator_certified_traces(
            &layout(&[("physical", 0), ("physical", 1)]),
            &[4, 9],
            &[4, 9],
            &[trace("physical", None, 2, Some(&[9, 4]))],
            &["physical".into()]
        )
        .unwrap_err()
        .to_string(),
        "runtime result trace \"physical\"[0] has measurement id 9, but the generator-certified layout requires 4"
    );
}

#[test]
fn generator_layout_allows_guppy_v1_array_without_scalar_ids() {
    let certified = generator_certified_traces(
        &layout(&[("physical", 0), ("physical", 1)]),
        &[4, 9],
        &[4, 9],
        &[trace("physical", None, 2, None)],
        &["physical".into()],
    )
    .unwrap();
    assert_eq!(
        certified,
        [scalar("physical", 0, 4), scalar("physical", 1, 9)]
    );
}

#[test]
fn generator_layout_checks_synthetic_array_slots_against_base_array_ids() {
    let entries = layout(&[("physical:meas:0", 0), ("physical:meas:1", 0)]);
    let traces = [
        trace("physical:meas:0", None, 1, None),
        trace("physical:meas:1", None, 1, None),
        trace("physical", None, 2, Some(&[9, 4])),
    ];
    assert_eq!(
        generator_certified_traces(
            &entries,
            &[4, 9],
            &[4, 9],
            &traces,
            &["physical:meas:0".into(), "physical:meas:1".into()]
        )
        .unwrap_err()
        .to_string(),
        "runtime result trace \"physical:meas:0\"[0] has measurement id 9, but the generator-certified layout requires 4"
    );
}

#[test]
fn generator_validation_order_and_messages() {
    let empty: &[ResultTrace] = &[];
    assert_eq!(
        generator_certified_traces(&[], &[0], &[0], empty, &[])
            .unwrap_err()
            .to_string(),
        "Guppy named measurement layout has 0 entries but the source trace has 1 measurements"
    );
    assert_eq!(
        generator_certified_traces(&layout(&[("a", 0), ("a", 0)]), &[0, 1], &[0, 1], empty, &[])
            .unwrap_err()
            .to_string(),
        "Guppy named measurement layout repeats output slot (\"a\", 0)"
    );
    assert_eq!(
        generator_certified_traces(&layout(&[("z", 0), ("a", 0)]), &[0, 1], &[0, 1], empty, &[])
            .unwrap_err()
            .to_string(),
        "Guppy named measurement layout expects 1 value(s) for \"z\", but the runtime trace emits 0"
    );
    assert_eq!(
        generator_certified_traces(
            &layout(&[("a", 0)]),
            &[0],
            &[0],
            &[scalar("a", 0, 0)],
            &["z".into(), "b".into(), "z".into()]
        )
        .unwrap_err()
        .to_string(),
        "result_ref tag(s) are absent from the generator-certified layout: [\"b\", \"z\"]"
    );
}

#[test]
fn compiler_raw_ids_agree_at_flattened_positions() {
    let bound = [trace("raw", None, 2, Some(&[3, 9]))];
    let wrong = [trace("raw", None, 2, Some(&[3, 7]))];
    for generator in [false, true] {
        let certified = if generator {
            vec![scalar("raw", 1, 9), scalar("raw", 0, 3)]
        } else {
            vec![trace("raw", Some(0), 2, Some(&[3, 9]))]
        };
        check_certified_raw_binding(&certified, &bound, "raw", generator).unwrap();
        assert_eq!(
            check_certified_raw_binding(&certified, &wrong, "raw", generator)
                .unwrap_err()
                .to_string(),
            "raw tag \"raw\" element 1: certified measurement ID 9 disagrees with inference binding ID Some(7)"
        );
        let unsupported = certified
            .into_iter()
            .map(|item| ResultTrace {
                measurement_ids: None,
                ..item
            })
            .collect::<Vec<_>>();
        check_certified_raw_binding(&unsupported, &bound, "raw", generator).unwrap();
    }
}

#[test]
fn certified_raw_ids_use_runtime_call_widths_and_generator_element_slots() {
    let bound = [
        trace("raw", None, 0, None),
        scalar("raw", 1, 3),
        scalar("raw", 2, 9),
    ];
    let compiler = [
        trace("raw", Some(0), 1, None),
        scalar("raw", 1, 3),
        scalar("raw", 2, 9),
    ];
    check_certified_raw_binding(&compiler, &bound, "raw", false).unwrap();
    let array = [trace("raw", None, 2, Some(&[3, 9]))];
    assert_eq!(
        check_certified_raw_binding(&[scalar("raw", 0, 3)], &array, "raw", false)
            .unwrap_err()
            .to_string(),
        "raw tag \"raw\" call 0 emits 2 value(s), but its certificate names 1 measurement(s)"
    );
    let generator = [scalar("raw:meas:1", 0, 9), scalar("raw:meas:0", 0, 3)];
    check_certified_raw_binding(&generator, &array, "raw", true).unwrap();
    assert_eq!(
        check_certified_raw_binding(
            &generator,
            &[trace("raw", None, 2, Some(&[3, 7]))],
            "raw",
            true
        )
        .unwrap_err()
        .to_string(),
        "raw tag \"raw\" element 1: certified measurement ID 9 disagrees with inference binding ID Some(7)"
    );
    assert_eq!(
        check_certified_raw_binding(&[scalar("raw", 9, 9)], &array, "raw", false)
            .unwrap_err()
            .to_string(),
        "raw tag \"raw\" element 2: certified measurement ID 9 disagrees with inference binding ID None"
    );
}

#[test]
fn mixed_build_checks_certified_raw_ids_against_inference() {
    let bound = [trace("raw measurements", None, 3, Some(&[2, 0, 1]))];
    let agrees = [trace("raw measurements", Some(0), 3, Some(&[2, 0, 1]))];
    check_certified_raw_binding(&agrees, &bound, "raw measurements", false).unwrap();
    let certified = [trace("raw measurements", Some(0), 3, Some(&[0, 1, 2]))];
    assert_eq!(
        check_certified_raw_binding(&certified, &bound, "raw measurements", false)
            .unwrap_err()
            .to_string(),
        "raw tag \"raw measurements\" element 0: certified measurement ID 0 disagrees with inference binding ID Some(2)"
    );
}

#[test]
fn ascii_synthetic_slots_and_exact_slot_precedence() {
    for text in ["1", "001"] {
        let name = format!("raw:meas:{text}");
        assert_eq!(
            synthetic_slot(&name),
            Some(SyntheticSlot {
                base: "raw",
                digits: text,
                element: Some(1)
            })
        );
        check_certified_raw_binding(
            &[scalar(&name, 0, 9)],
            &[trace("raw", None, 2, Some(&[3, 9]))],
            "raw",
            true,
        )
        .unwrap();
    }
    for text in ["", "-1", "²", "1x", "١", "１", "𝟙"] {
        let name = format!("raw:meas:{text}");
        assert_eq!(synthetic_slot(&name), None);
        check_certified_raw_binding(&[scalar(&name, 0, 9)], &[], "raw", true).unwrap();
    }
    let entries = layout(&[("raw:meas:0", 0)]);
    generator_certified_traces(
        &entries,
        &[4],
        &[4],
        &[scalar("raw:meas:0", 0, 4), scalar("raw", 0, 9)],
        &[],
    )
    .unwrap();
}

#[test]
fn synthetic_slot_overflow_is_a_missing_binding_not_an_ignored_tag() {
    let number = "184467440737095516160";
    let certified = [scalar(&format!("raw:meas:{number}"), 0, 9)];
    assert_eq!(
        check_certified_raw_binding(&certified, &[], "raw", true)
            .unwrap_err()
            .to_string(),
        format!(
            "raw tag \"raw\" element {number:?}: certified measurement ID 9 disagrees with inference binding ID None"
        )
    );
    let past_end = [scalar("raw:meas:1", 0, 9)];
    assert_eq!(
        check_certified_raw_binding(&past_end, &[], "raw", true)
            .unwrap_err()
            .to_string(),
        "raw tag \"raw\" element 1: certified measurement ID 9 disagrees with inference binding ID None"
    );
    for name in [format!("raw:meas:{number}"), "raw:meas:1".into()] {
        // A missing optional runtime identity does not contradict the certificate.
        let entries = layout(&[(&name, 0)]);
        let traces = [trace(&name, None, 1, None), scalar("raw", 0, 4)];
        let certified = generator_certified_traces(&entries, &[9], &[9], &traces, &[]).unwrap();
        assert_eq!(certified, [scalar(&name, 0, 9)]);
    }
}

#[test]
fn maximum_layout_slot_reports_width_and_raw_binding_mismatch() {
    assert_eq!(
        generator_certified_traces(&layout(&[("raw", usize::MAX)]), &[0], &[0], &[], &[])
            .unwrap_err()
            .to_string(),
        format!(
            "Guppy named measurement layout expects {} value(s) for \"raw\", but the runtime trace emits 0",
            usize::MAX as u128 + 1
        )
    );
    assert_eq!(
        check_certified_raw_binding(&[scalar("raw", usize::MAX, 9)], &[], "raw", true)
            .unwrap_err()
            .to_string(),
        format!(
            "raw tag \"raw\" element {}: certified measurement ID 9 disagrees with inference binding ID None",
            usize::MAX
        )
    );
}

#[test]
fn rec_build_evaluates_compiler_certified_scalar_results() {
    let traces = compiler_certified_traces(
        &[0, 1],
        &[1, 0],
        &[scalar("a", 0, 0), scalar("b", 0, 1)],
        &[],
        || Ok(analysis()),
    )
    .unwrap();
    let index = ResultTraceIndex::new(&[1, 0], &traces).unwrap();
    assert_eq!(
        index
            .resolve_ref(&MeasurementRef::record(2).unwrap())
            .unwrap(),
        0
    );
    assert_eq!(
        index
            .resolve_ref(&MeasurementRef::record(1).unwrap())
            .unwrap(),
        1
    );
    assert_eq!(
        named_binding_classification(None, false, &index.named_bindings(None)),
        "compiler_direct_scalar_complete"
    );
}

#[test]
fn rec_build_rejects_transformed_scalar_result_as_measurement_provenance() {
    let runtime = [trace("not_m", None, 1, None)];
    let traces = compiler_certified_traces(&[0], &[0], &runtime, &[], || {
        Ok((BTreeMap::from([("not_m".into(), vec![None])]), 1))
    })
    .unwrap();
    let bindings = ResultTraceIndex::new(&[0], &traces)
        .unwrap()
        .named_bindings(None);
    assert_eq!(bindings["not_m"], [None]);
    assert_eq!(named_binding_classification(None, false, &bindings), "none");
}

#[test]
fn rec_build_rejects_aggregate_array_as_measurement_provenance() {
    let runtime = [trace("pair", None, 2, Some(&[0, 1]))];
    let traces = compiler_certified_traces(&[0, 1], &[0, 1], &runtime, &[], || {
        Ok((BTreeMap::from([("pair".into(), vec![None])]), 2))
    })
    .unwrap();
    assert_eq!(
        ResultTraceIndex::new(&[0, 1], &traces)
            .unwrap()
            .named_bindings(None)["pair"],
        [None, None]
    );
}

#[test]
fn generator_certificate_rehash_still_checks_runtime_identity() {
    let entries = layout(&[("sx1:init:meas:1", 0), ("sx1:init:meas:0", 0)]);
    let traces = [
        scalar("sx1:init:meas:0", 0, 0),
        scalar("sx1:init:meas:1", 0, 1),
    ];
    assert_eq!(
        generator_certified_traces(&entries, &[0, 1], &[0, 1], &traces, &[])
            .unwrap_err()
            .to_string(),
        "runtime result trace \"sx1:init:meas:1\"[0] has measurement id 1, but the generator-certified layout requires 0"
    );
}

#[test]
fn empty_runtime_tag_is_valid_only_for_unsupported_calls() {
    assert!(ResultTraceIndex::new(&[0], &[trace("", None, 1, None)]).is_ok());
    assert_eq!(
        ResultTraceIndex::new(&[0], &[scalar("", 0, 0)])
            .unwrap_err()
            .to_string(),
        "result_ref tag must be a non-empty string"
    );
}

#[test]
fn sparse_generator_slots_remain_sparse_named_bindings() {
    let certified = generator_certified_traces(
        &layout(&[("physical", 1)]),
        &[9],
        &[9],
        &[trace("physical", None, 2, None)],
        &[],
    )
    .unwrap();
    assert_eq!(certified, [scalar("physical", 1, 9)]);
    assert!(
        ResultTraceIndex::new(&[9], &certified)
            .unwrap()
            .named_bindings(None)
            .is_empty()
    );
}

#[test]
fn raw_binding_check_rejects_unnumbered_certified_calls() {
    let raw = [trace("raw", None, 1, Some(&[3]))];
    assert_eq!(
        check_certified_raw_binding(&raw, &raw, "raw", false)
            .unwrap_err()
            .to_string(),
        "certified result trace \"raw\" has no explicit occurrence"
    );
}
