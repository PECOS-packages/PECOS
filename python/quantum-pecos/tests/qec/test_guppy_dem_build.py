# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Adversarial measurement-identity tests for typed Guppy DEM builds."""

from __future__ import annotations

import hashlib
import json
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pecos
import pecos.tracing as tracing
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, result
from guppylang.std.quantum import cx, h, measure, qubit, x
from pecos._compilation import guppy_to_hugr
from pecos._qis_trace_replay import (
    _replay_qis_trace_chunks_into_tick_circuit,
    _validate_audited_trace_stream,
)
from pecos.guppy_gen import get_num_qubits, make_surface_code
from pecos.qec import (
    Detector,
    DetectorErrorModel,
    Observable,
    _hugr_preflight,
    _qis_result_tags,
    build_dem_from_guppy,
    rec,
    result_ref,
    surface_memory_dem_spec,
)
from pecos.qec._qis_result_tags import _CertificationError
from pecos.qec.dem import _generator_certified_result_traces, _GuppyProgramAnalysis
from pecos.qec.dem_spec import GuppyDemBuild, RecordRef, ResultRef, _resolve_dem_specs
from pecos_rslib.quantum import TickCircuit

from qec._measurement_expectations import measurement_layout


@guppy
def _scrambled_tagged_measurements() -> None:
    qa = qubit()
    qb = qubit()
    h(qb)
    cx(qb, qa)
    a = measure(qa).read()
    b = measure(qb).read()
    result("b", b)
    result("a", a)


@guppy
def _transformed_measurement_result() -> None:
    q = qubit()
    measured = measure(q).read()
    result("not_m", not measured)


@guppy
def _aggregate_measurement_result() -> None:
    qa = qubit()
    qb = qubit()
    a = measure(qa).read()
    b = measure(qb).read()
    result("pair", array(a, b))


@guppy
def _mixed_supported_result_occurrences() -> None:
    q = qubit()
    measured = measure(q).read()
    result("same", not measured)
    result("same", measured)


@guppy
def _measurement_feedback_without_named_results() -> None:
    q0 = qubit()
    q1 = qubit()
    h(q0)
    if measure(q0).read():
        x(q1)
    _ = measure(q1).read()


@guppy
def _measurement_without_named_results() -> None:
    q = qubit()
    _ = measure(q).read()


def _unexpected_program_analysis(*_args: object) -> None:
    msg = "Unused program analysis must remain lazy"
    raise AssertionError(msg)


@pytest.mark.parametrize("wrapped", [False, True])
def test_certified_build_does_not_load_hugr(monkeypatch: pytest.MonkeyPatch, *, wrapped: bool) -> None:
    """Certified builds skip the HUGR load and analyze the lowered text exactly once.

    The layout is a claim checked against independent evidence, so the static
    analysis now runs; the control-flow admission still needs no HUGR load.
    """
    monkeypatch.setattr(_hugr_preflight, "load_hugr_from_bytes", _unexpected_program_analysis)
    extractor = Mock(wraps=_qis_result_tags.extract_result_tag_measurements)
    monkeypatch.setattr(_qis_result_tags, "extract_result_tag_measurements", extractor)
    program = make_surface_code(3, 1, "Z")
    detectors, observables = surface_memory_dem_spec(3, 1, "Z")
    build = build_dem_from_guppy(
        pecos.Guppy(program) if wrapped else program,
        num_qubits=get_num_qubits(3),
        detectors=detectors,
        observables=observables,
        p_meas=0.1,
    )
    assert build.audit["named_result_binding"] == "generator_layout_v2_program_bound"
    assert build.dem.num_detectors == len(detectors)
    extractor.assert_called_once()


@pytest.mark.parametrize("references", ["typed", "json"])
def test_tagged_build_loads_hugr_and_analyzes_qis_once(monkeypatch: pytest.MonkeyPatch, references: str) -> None:
    """Regression: packet A shares lazy analysis across preflight and tag consumers."""
    loader = Mock(wraps=_hugr_preflight.load_hugr_from_bytes)
    extractor = Mock(wraps=_qis_result_tags.extract_result_tag_measurements)
    monkeypatch.setattr(_hugr_preflight, "load_hugr_from_bytes", loader)
    monkeypatch.setattr(_qis_result_tags, "extract_result_tag_measurements", extractor)
    if references == "typed":
        dem = build_dem_from_guppy(
            _scrambled_tagged_measurements,
            num_qubits=2,
            detectors=[Detector(result_ref("a"))],
            p_meas=0.1,
        ).dem
    else:
        dem = DetectorErrorModel.from_guppy(
            _scrambled_tagged_measurements,
            num_qubits=2,
            detectors_json='[{"id":0,"result_tags":["a"]}]',
            p_meas=0.1,
        )
    assert dem.num_detectors == 1
    assert loader.call_count == extractor.call_count == 1


def test_build_without_candidate_tags_skips_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: packet A must preserve the no-candidate-tags early return."""
    loader = Mock(wraps=_hugr_preflight.load_hugr_from_bytes)
    monkeypatch.setattr(_hugr_preflight, "load_hugr_from_bytes", loader)
    monkeypatch.setattr(_qis_result_tags, "extract_result_tag_measurements", _unexpected_program_analysis)
    build = build_dem_from_guppy(
        _measurement_without_named_results,
        num_qubits=1,
        detectors=[],
        p_meas=0.1,
    )
    assert build.dem.num_detectors == 0
    assert loader.call_count == 1


def _reordered_trace() -> tuple[TickCircuit, list[dict[str, object]]]:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([2, 0, 1], [2, 0, 1])
    traces = [
        {"name": "a", "values": [False], "result_ids": [0]},
        {"name": "b", "values": [False], "result_ids": [1]},
        {"name": "c", "values": [False], "result_ids": [2]},
    ]
    return circuit, traces


def test_named_refs_follow_identities_after_runtime_reorder() -> None:
    circuit, traces = _reordered_trace()
    schema = _resolve_dem_specs(
        [Detector(result_ref("a"), result_ref("c"))],
        [Observable(result_ref("b"))],
        circuit=circuit,
        result_traces=traces,
    )
    assert json.loads(schema.detectors_json) == [{"id": 0, "meas_ids": [0, 2]}]
    assert json.loads(schema.observables_json) == [{"id": 0, "meas_ids": [1]}]
    assert [entry.meas_id for entry in schema.ledger] == [2, 0, 1]


def test_named_refs_match_verified_runtime_json_offsets() -> None:
    _, offsets = measurement_layout(_scrambled_tagged_measurements, 2)
    noise = {"p1": 0.01, "p2": 0.02, "p_meas": 0.1, "p_prep": 0.0}
    via_records = DetectorErrorModel.from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=2,
        detectors_json=json.dumps([{"id": 0, "records": [offsets[0]]}]),
        observables_json=json.dumps([{"id": 0, "records": [offsets[1]]}]),
        **noise,
    )
    via_results = build_dem_from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=2,
        detectors=[Detector(result_ref("a"))],
        observables=[Observable(result_ref("b"))],
        **noise,
    )
    assert via_records.to_string() == via_results.dem.to_string()


def test_bare_tag_strings_are_shorthand_for_result_ref() -> None:
    noise = {"p1": 0.01, "p2": 0.02, "p_meas": 0.1, "p_prep": 0.0}
    via_result_ref = build_dem_from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=2,
        detectors=[Detector(result_ref("a"))],
        observables=[Observable(result_ref("b"))],
        **noise,
    )
    via_strings = build_dem_from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=2,
        detectors=[Detector("a")],
        observables=[Observable("b")],
        **noise,
    )

    assert via_strings.detectors_json == via_result_ref.detectors_json
    assert via_strings.observables_json == via_result_ref.observables_json
    assert via_strings.schema_fingerprint == via_result_ref.schema_fingerprint
    assert via_strings.dem.to_string() == via_result_ref.dem.to_string()


def test_tag_strings_mix_with_rec_and_result_ref_refs() -> None:
    detector = Detector("a", result_ref("b"), rec[-1])

    assert detector.refs == (ResultRef("a"), ResultRef("b"), RecordRef(-1))


def test_tag_string_shorthand_rejects_empty_and_wrong_types() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        Detector("")
    with pytest.raises(TypeError, match="measurement references must be"):
        Detector(3.5)
    with pytest.raises(ValueError, match="at least one measurement"):
        Observable()


def test_trace_once_build_evaluates_runtime_and_rejects_uncertified_named_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_trace(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        circuit, traces = _reordered_trace()
        traces.append({"name": "uncertified", "values": [False], "result_ids": [2]})
        return circuit, traces

    monkeypatch.setattr(
        "pecos.tracing._trace_program_to_tick_circuit_with_result_traces",
        fake_trace,
    )
    monkeypatch.setattr(
        _GuppyProgramAnalysis,
        "result_tags",
        property(lambda _: ({"a": [0], "b": [1], "c": [2], "uncertified": [None]}, 3)),
    )
    build = build_dem_from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=3,
        detectors=[
            Detector(result_ref("a"), result_ref("c")),
            Detector(result_ref("a"), result_ref("c")),
        ],
        observables=[Observable(result_ref("b"))],
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
    )

    assert calls == 1
    assert build.audit["runtime_measurement_order"] == [2, 0, 1]
    assert build.audit["runtime_order_mismatch_count"] == 3
    assert build.evaluate_runtime_record([1, 0, 1]) == ([1, 1], 1)
    with pytest.raises(ValueError, match="missing referenced MeasId"):
        build.evaluate_results({"a": [0], "b": [1], "uncertified": [1]})


def test_named_build_evaluates_compiler_certified_scalar_results() -> None:
    build = build_dem_from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=2,
        detectors=[Detector(result_ref("a"))],
        observables=[Observable(result_ref("b"))],
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
    )

    assert build.audit["named_result_binding"] == "compiler_direct_scalar_complete"
    assert build.evaluate_results({"a": [0], "b": [1]}) == ([0], 1)


def test_typed_build_rejects_dynamic_control_without_named_results() -> None:
    with pytest.raises(ValueError, match="branching or looping control flow"):
        build_dem_from_guppy(
            _measurement_feedback_without_named_results,
            num_qubits=2,
            detectors=[Detector(result_ref("b"), result_ref("a"))],
            p1=0.0,
            p2=0.0,
            p_meas=0.1,
            p_prep=0.0,
        )


def test_uncertifiable_inputs_fail_closed_on_both_entry_points() -> None:
    """A program whose HUGR cannot be obtained must be rejected, not traced:
    one sampled execution of an uninspectable program is not a static circuit.
    """
    from pecos.qec import DetectorErrorModel

    with pytest.raises(ValueError, match="HUGR-compilable"):
        build_dem_from_guppy(
            object(),
            num_qubits=1,
            detectors=[Detector(result_ref("a"))],
            p1=0.0,
            p2=0.0,
            p_meas=0.1,
            p_prep=0.0,
        )
    with pytest.raises(ValueError, match="HUGR-certifiable"):
        DetectorErrorModel.from_guppy(
            object(),
            num_qubits=1,
            detectors_json='[{"id":0,"records":[-1]}]',
            p1=0.0,
            p2=0.0,
            p_meas=0.1,
            p_prep=0.0,
        )


def test_dynamic_hugr_bytes_wrapper_is_rejected_not_traced() -> None:
    """Compiled dynamic control flow must not dodge the guard by arriving as
    a pecos.Hugr wrapper or raw HUGR envelope bytes."""
    import pecos
    from pecos._compilation import guppy_to_hugr

    dynamic_bytes = guppy_to_hugr(_measurement_feedback_without_named_results)
    for program in (pecos.Hugr(dynamic_bytes), dynamic_bytes):
        with pytest.raises(ValueError, match="branching or looping control flow"):
            build_dem_from_guppy(
                program,
                num_qubits=2,
                detectors=[Detector(result_ref("b"), result_ref("a"))],
                p1=0.0,
                p2=0.0,
                p_meas=0.1,
                p_prep=0.0,
            )


def test_all_certifiable_input_forms_build_identical_dems() -> None:
    """@guppy def, pecos.Guppy, pecos.Hugr, and raw bytes are all accepted
    and certify/trace the same HUGR, so the DEMs are byte-identical."""
    import pecos
    from pecos._compilation import guppy_to_hugr

    hugr_bytes = guppy_to_hugr(_scrambled_tagged_measurements)
    noise = {"p1": 0.01, "p2": 0.02, "p_meas": 0.1, "p_prep": 0.0}
    dems = [
        build_dem_from_guppy(
            program,
            num_qubits=2,
            detectors=[Detector(result_ref("b"))],
            observables=[Observable(result_ref("a"))],
            **noise,
        ).dem.to_string()
        for program in (
            _scrambled_tagged_measurements,
            pecos.Guppy(_scrambled_tagged_measurements),
            pecos.Hugr(hugr_bytes),
            hugr_bytes,
        )
    ]
    assert len(set(dems)) == 1


def test_guppy_wrapped_generator_keeps_its_certificate() -> None:
    """pecos.Guppy(make_surface_code(...)) must still take the generator
    certificate path (the certificate lives on the wrapped definition)."""
    import pecos

    program = make_surface_code(3, 1, "Z")
    detectors, observables = surface_memory_dem_spec(3, 1, "Z")
    build = build_dem_from_guppy(
        pecos.Guppy(program),
        num_qubits=get_num_qubits(3),
        detectors=detectors,
        observables=observables,
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
    )

    assert build.audit["named_result_binding"] == "generator_layout_v2_program_bound"


def test_certified_definition_does_not_certify_foreign_bytes() -> None:
    """A pecos.Guppy whose envelope differs from its certified generator
    definition must fail the digest check: the certificate binds the bytes
    that are audited and executed, not the definition it was stamped on."""
    import pecos
    from pecos._compilation import guppy_to_hugr

    foreign_bytes = guppy_to_hugr(_scrambled_tagged_measurements)

    class _MismatchedGuppy(pecos.Guppy):
        @property
        def hugr_bytes(self) -> bytes:
            return foreign_bytes

    detectors, observables = surface_memory_dem_spec(3, 1, "Z")
    with pytest.raises(ValueError, match="certificate does not match the program and layout"):
        build_dem_from_guppy(
            _MismatchedGuppy(make_surface_code(3, 1, "Z")),
            num_qubits=get_num_qubits(3),
            detectors=detectors,
            observables=observables,
            p1=0.0,
            p2=0.0,
            p_meas=0.1,
            p_prep=0.0,
        )


@pytest.mark.parametrize("claimed_definition", ["none", "self", "stamped_object"])
def test_forged_certificate_on_byte_carrier_does_not_bypass_the_guard(claimed_definition: str) -> None:
    """A self-consistent digest stapled to a byte carrier must not suppress
    the control-flow guard: certificates are honored only on Guppy definition
    objects, never on byte carriers, whatever ``wrapped_function`` they claim."""
    import hashlib as _hashlib

    import pecos
    from pecos._compilation import guppy_to_hugr

    dynamic_bytes = guppy_to_hugr(_measurement_feedback_without_named_results)
    layout_json = json.dumps([], separators=(",", ":"))
    digest = _hashlib.sha256(dynamic_bytes + b"\0" + layout_json.encode()).hexdigest()
    forged = pecos.Hugr(dynamic_bytes)
    forged.__pecos_named_measurement_layout_v2__ = (digest, [])
    if claimed_definition == "self":
        forged.wrapped_function = forged
    elif claimed_definition == "stamped_object":

        class _Stamped:
            __pecos_named_measurement_layout_v2__ = (digest, [])

        forged.wrapped_function = _Stamped()

    with pytest.raises(ValueError, match="branching or looping control flow"):
        build_dem_from_guppy(
            forged,
            num_qubits=2,
            detectors=[Detector(result_ref("b"), result_ref("a"))],
            p1=0.0,
            p2=0.0,
            p_meas=0.1,
            p_prep=0.0,
        )


@pytest.mark.parametrize("corruption", ["duplicate_terminal", "nonempty_terminal"])
def test_audited_trace_stream_rejects_malformed_terminals(corruption: str) -> None:
    chunks = _framed_trace_chunks()
    if corruption == "duplicate_terminal":
        chunks.insert(
            1,
            {
                **chunks[0],
                "chunk_index": 1,
                "stage": "trace_complete",
                "num_operations": 0,
                "operations": [],
            },
        )
        chunks[2] = {**chunks[2], "chunk_index": 2}
    else:
        chunks[-1] = {
            **chunks[-1],
            "num_operations": 1,
            "operations": [{"Quantum": {"H": 0}}],
        }

    with pytest.raises(ValueError, match=r"exactly one terminal|terminal chunk must be empty"):
        _validate_audited_trace_stream(chunks)


def test_spec_rejects_non_json_metadata_and_dead_parity() -> None:
    circuit, traces = _reordered_trace()

    with pytest.raises(ValueError, match="JSON-serializable"):
        _resolve_dem_specs(
            [Detector(result_ref("c"), metadata={"bad": {1, 2}})],
            [],
            circuit=circuit,
            result_traces=traces,
        )
    with pytest.raises(ValueError, match="JSON-serializable"):
        _resolve_dem_specs(
            [Detector(result_ref("c"), metadata={"bad": float("nan")})],
            [],
            circuit=circuit,
            result_traces=traces,
        )
    with pytest.raises(ValueError, match="even multiplicity"):
        _resolve_dem_specs(
            [Detector(result_ref("c"), result_ref("c"))],
            [],
            circuit=circuit,
            result_traces=traces,
        )


def test_dynamic_control_is_rejected_before_any_trace_executes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pin the preflight ordering: rejection must happen without tracing.

    If the control-flow guard ever moves back behind the runtime trace, the
    monkeypatched trace raises AssertionError instead of the expected
    ValueError and this test fails.
    """

    def _trace_must_not_run(*_args: object, **_kwargs: object) -> None:
        msg = "trace executed before the static-schedule preflight"
        raise AssertionError(msg)

    monkeypatch.setattr(
        "pecos.tracing._trace_program_to_tick_circuit_with_result_traces",
        _trace_must_not_run,
    )
    with pytest.raises(ValueError, match="branching or looping control flow"):
        build_dem_from_guppy(
            _measurement_feedback_without_named_results,
            num_qubits=2,
            detectors=[Detector(result_ref("b"), result_ref("a"))],
            p1=0.0,
            p2=0.0,
            p_meas=0.1,
            p_prep=0.0,
        )


def test_named_build_rejects_transformed_scalar_result_as_measurement_provenance() -> None:
    with pytest.raises(ValueError, match="not a direct scalar measurement"):
        build_dem_from_guppy(_transformed_measurement_result, num_qubits=1, detectors=[Detector(result_ref("not_m"))])


def test_shot_evaluators_reject_non_bit_values() -> None:
    circuit, traces = _reordered_trace()
    schema = _resolve_dem_specs(
        [Detector(result_ref("a"))],
        [],
        circuit=circuit,
        result_traces=traces,
    )
    build = GuppyDemBuild(
        dem=None,
        circuit=circuit,
        detectors_json=schema.detectors_json,
        observables_json=schema.observables_json,
        measurement_ledger=schema.ledger,
        schema_fingerprint=schema.schema_fingerprint,
        named_result_binding="compiler_direct_scalar_complete",
        _detector_meas_ids=schema.detector_meas_ids,
        _observable_meas_ids=schema.observable_meas_ids,
        _result_ids_by_tag=schema.result_ids_by_tag,
    )

    with pytest.raises(ValueError, match="must be bool, 0, or 1"):
        build.evaluate_runtime_record([0, 2, 0])
    with pytest.raises(ValueError, match="must be bool, 0, or 1"):
        build.evaluate_results({"a": ["0"], "b": [0], "c": [0]})
    with pytest.raises(ValueError, match="must be bool, 0, or 1"):
        build.evaluate_measurements({0: 2})


def test_result_ref_rejects_non_string_tag() -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        result_ref(123)  # type: ignore[arg-type]


def test_named_build_rejects_aggregate_array_as_measurement_provenance() -> None:
    with pytest.raises(ValueError, match="not a direct scalar measurement"):
        build_dem_from_guppy(_aggregate_measurement_result, num_qubits=2, detectors=[Detector(result_ref("pair"))])


def test_result_ref_rejects_ambiguous_or_missing_runtime_provenance() -> None:
    circuit, _ = _reordered_trace()
    ambiguous = [{"name": "computed", "values": [True], "result_ids": []}]

    with pytest.raises(ValueError, match="not a direct scalar measurement"):
        _resolve_dem_specs(
            [Detector(result_ref("computed"))],
            [],
            circuit=circuit,
            result_traces=ambiguous,
        )


def test_result_ref_preserves_unsupported_occurrence_holes() -> None:
    with pytest.raises(ValueError, match=r"occurrence 0.*not a direct scalar measurement"):
        build_dem_from_guppy(
            _mixed_supported_result_occurrences,
            num_qubits=1,
            detectors=[Detector(result_ref("same", occurrence=0))],
            p1=0.0,
            p2=0.0,
            p_meas=0.1,
            p_prep=0.0,
        )

    build = build_dem_from_guppy(
        _mixed_supported_result_occurrences,
        num_qubits=1,
        detectors=[Detector(result_ref("same", occurrence=1))],
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
    )
    assert build.evaluate_results({"same": [True, False]}) == ([0], 0)


def test_result_ref_rejects_array_element_provenance() -> None:
    circuit, _ = _reordered_trace()
    array_trace = [{"name": "pair", "values": [False, False], "result_ids": [0, 1]}]

    with pytest.raises(ValueError, match="array-valued result_ref provenance"):
        _resolve_dem_specs(
            [Detector(result_ref("pair", element=0))],
            [],
            circuit=circuit,
            result_traces=array_trace,
        )


def test_scalar_result_provenance_takes_precedence_over_aggregate_arrays() -> None:
    circuit, _ = _reordered_trace()
    traces = [
        {"name": "a", "values": [False], "result_ids": [0]},
        {"name": "b", "values": [False], "result_ids": [1]},
        {"name": "aggregate", "values": [False, False], "result_ids": [1, 0]},
        {"name": "final", "values": [False], "result_ids": [2]},
    ]
    schema = _resolve_dem_specs(
        [Detector(result_ref("a"), result_ref("final"))],
        [],
        circuit=circuit,
        result_traces=traces,
    )

    assert dict(schema.result_ids_by_tag) == {"a": (0,), "b": (1,), "final": (2,)}


def test_resolver_defensively_rejects_record_refs() -> None:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [4, 9])

    with pytest.raises(TypeError, match=r"detector 0 uses rec\[-1\].*do not preserve source measurement order"):
        _resolve_dem_specs(
            [Detector(rec[-1])],
            [],
            circuit=circuit,
            result_traces=[],
        )


def test_result_ref_supports_non_dense_runtime_measurement_identities() -> None:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [4, 9])
    traces = [
        {"name": "first", "occurrence": 0, "values": [False], "result_ids": [4]},
        {"name": "second", "occurrence": 0, "values": [False], "result_ids": [9]},
    ]

    schema = _resolve_dem_specs(
        [Detector(result_ref("second"))],
        [],
        circuit=circuit,
        result_traces=traces,
    )

    assert json.loads(schema.detectors_json) == [{"id": 0, "meas_ids": [9]}]


def test_audited_build_disables_raw_measurement_id_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_trace(*_args, **kwargs):
        seen.update(kwargs)
        return _reordered_trace()

    monkeypatch.setattr(
        "pecos.tracing._trace_program_to_tick_circuit_with_result_traces",
        fake_trace,
    )
    monkeypatch.setattr(_GuppyProgramAnalysis, "result_tags", property(lambda _: ({"a": [0], "b": [1], "c": [2]}, 3)))
    build_dem_from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=3,
        detectors=[Detector(result_ref("a"))],
        p1=0.0,
        p2=0.0,
        p_meas=0.0,
        p_prep=0.0,
    )

    assert seen["allow_raw_measurement_id_fallback"] is False


def test_audited_replay_rejects_entirely_raw_runtime_trace() -> None:
    chunks = [
        {
            "operations": [
                {"AllocateQubit": {"id": 0}},
                {"Quantum": {"Measure": [0, 0]}},
            ],
        },
    ]

    with pytest.raises(ValueError, match="does not contain lowered_quantum_ops"):
        _replay_qis_trace_chunks_into_tick_circuit(
            chunks,
            allow_raw_measurement_id_fallback=False,
        )


def _framed_trace_chunks() -> list[dict[str, object]]:
    common = {
        "format": "pecos_qis_operation_trace_v1",
        "engine_trace_id": 7,
        "shot_index": 1,
    }
    return [
        {
            **common,
            "chunk_index": 0,
            "stage": "pending_final",
            "num_operations": 1,
            "operations": [{"Quantum": {"Measure": [0, 0]}}],
        },
        {
            **common,
            "chunk_index": 1,
            "stage": "trace_complete",
            "num_operations": 0,
            "operations": [],
        },
    ]


def test_audited_trace_stream_accepts_contiguous_terminal_framing() -> None:
    _validate_audited_trace_stream(_framed_trace_chunks())


@pytest.mark.parametrize("corruption", ["gap", "cross_shot", "truncated"])
def test_audited_trace_stream_rejects_incomplete_or_mixed_framing(corruption: str) -> None:
    chunks = _framed_trace_chunks()
    if corruption == "gap":
        chunks[1]["chunk_index"] = 2
    elif corruption == "cross_shot":
        chunks[1]["shot_index"] = 2
    else:
        chunks.pop()

    with pytest.raises(ValueError, match=r"contiguous|mixes engine or shot|terminal"):
        _validate_audited_trace_stream(chunks)


def test_generated_surface_named_results_evaluate_without_private_adapter() -> None:
    rounds = 2
    program = make_surface_code(3, rounds, "Z")
    detectors, observables = surface_memory_dem_spec(3, rounds, "Z")
    build = build_dem_from_guppy(
        program,
        num_qubits=get_num_qubits(3),
        detectors=detectors,
        observables=observables,
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
    )
    columns = (
        pecos.sim(program)
        .classical(pecos.selene_engine())
        .quantum(pecos.stabilizer())
        .qubits(get_num_qubits(3))
        .seed(7)
        .run(3)
        .to_shot_map()
        .to_dict()
    )

    evaluated = build.evaluate_result_columns(columns)

    assert build.audit["named_result_binding"] == "generator_layout_v2_program_bound"
    assert len(evaluated) == 3
    assert all(len(events) == build.dem.num_detectors for events, _ in evaluated)
    assert evaluated == [([0] * build.dem.num_detectors, 0)] * 3


class _UnavailableResultAnalysis:
    @property
    def result_tags(self) -> tuple[dict[str, list[int | None]], int]:
        msg = "test schedule has no static evidence"
        raise _CertificationError(msg)


@pytest.fixture
def strip_runtime_result_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    original = tracing.named_result_traces_from_operation_trace

    def stripped(chunks: list[dict]) -> list[dict]:
        return [{**trace, "result_ids": []} for trace in original(chunks)]

    monkeypatch.setattr(tracing, "named_result_traces_from_operation_trace", stripped)


@pytest.mark.usefixtures("strip_runtime_result_ids")
@pytest.mark.parametrize("reordered", [False, True])
def test_generator_layout_uses_compiled_evidence_without_runtime_ids(*, reordered: bool) -> None:
    @guppy
    def program() -> None:
        qa = qubit()
        qb = qubit()
        qc = qubit()
        x(qb)
        a = measure(qa).read()
        b = measure(qb).read()
        c = measure(qc).read()
        result("a", a)
        result("b", b)
        result("c", c)

    hugr_bytes = guppy_to_hugr(program)
    occurrences, static_count = _qis_result_tags.extract_result_tag_measurements(_GuppyProgramAnalysis(hugr_bytes).qis)
    ids, offsets = measurement_layout(program, 3)
    assert occurrences == {tag: [offsets[q] + 3] for q, tag in enumerate("abc")}
    assert static_count == 3
    right = build_dem_from_guppy(program, num_qubits=3, detectors=[Detector(result_ref("b"))])
    wrong = build_dem_from_guppy(program, num_qubits=3, detectors=[Detector(result_ref("c"))])
    # Default p1 makes the extra X on b distinguish the physical noise sites.
    assert right.dem.to_string() != wrong.dem.to_string()
    expected_ids = {tag: ids[q] for q, tag in enumerate("abc")}
    layout = (("a", 0), ("c", 0), ("b", 0)) if reordered else (("a", 0), ("b", 0), ("c", 0))
    layout_json = json.dumps(layout, separators=(",", ":"))
    digest = hashlib.sha256(hugr_bytes + b"\0" + layout_json.encode()).hexdigest()
    object.__setattr__(program, "__pecos_named_measurement_layout_v2__", (digest, layout))
    built = build_dem_from_guppy(program, num_qubits=3, detectors=[Detector(result_ref("b"))])
    assert json.loads(built.detectors_json) == [{"id": 0, "meas_ids": [expected_ids["b"]]}]
    assert {ref.tag: entry.meas_id for entry in built.measurement_ledger for ref in entry.result_refs} == expected_ids
    assert built.dem.to_string() == right.dem.to_string()
    assert built.dem.to_string() != wrong.dem.to_string()
    for tag, identity in expected_ids.items():
        assert built.evaluate_results({tag: 1, **dict.fromkeys(expected_ids.keys() - {tag}, 0)}) == (
            [int(identity == expected_ids["b"])],
            0,
        )


@pytest.mark.usefixtures("strip_runtime_result_ids")
def test_generator_surface_without_any_evidence_rejects_build() -> None:
    program = make_surface_code(3, 1, "Z")
    analysis = _GuppyProgramAnalysis(guppy_to_hugr(program))
    with pytest.raises(_CertificationError, match="CFG cycle"):
        _qis_result_tags.extract_result_tag_measurements(analysis.qis)
    first_tag, first_index = program.__pecos_named_measurement_layout_v2__[1][0]
    error = (
        rf"{first_tag!r}\[{first_index}\]: "
        "neither runtime provenance nor compiled analysis identifies its measurement"
    )
    with pytest.raises(ValueError, match=error):
        build_dem_from_guppy(program, num_qubits=get_num_qubits(3), detectors=[Detector(result_ref("final:meas:0"))])


@pytest.mark.parametrize("evidence", ["runtime", "static", "both"])
def test_generator_scalar_evidence_binds_repeated_tag_occurrences(evidence: str) -> None:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1, 2], [4, 9, 12])
    circuit.set_meta("qis_source_measurement_ids", "[4,9,12]")
    layout = (("same", 0), ("other", 0), ("same", 1))
    analysis = (
        _UnavailableResultAnalysis()
        if evidence == "runtime"
        else SimpleNamespace(result_tags=({"same": [2, 0], "other": [1]}, 3))
    )
    traces = [
        {"name": tag, "values": [False], "result_ids": [] if evidence == "static" else [identity]}
        for tag, identity in [("same", 12), ("other", 9), ("same", 4)]
    ]
    assert _generator_certified_result_traces(layout, analysis, circuit, traces, required_tags=["same"]) == [
        {"name": tag, "occurrence": index, "values": [False], "result_ids": [identity]}
        for (tag, index), identity in zip(layout, [12, 9, 4], strict=True)
    ]


@pytest.mark.parametrize("evidence", ["runtime", "static", "mixed"])
@pytest.mark.parametrize("reverse_layout", [False, True])
@pytest.mark.parametrize("same_tag", [False, True])
def test_generator_slots_require_distinct_measurement_ids(
    evidence: str,
    reverse_layout: bool,
    same_tag: bool,
) -> None:
    """All slots must bind distinct IDs, even without any referenced tags."""
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [4, 9])
    circuit.set_meta("qis_source_measurement_ids", "[4,9]")
    layout = (("a", 0), ("a", 1) if same_tag else ("b", 0))
    occurrences = {"a": [0, 0]} if same_tag else {"a": [0], "b": [0]}
    if evidence == "mixed":
        occurrences["a"][0] = None
    analysis = _UnavailableResultAnalysis() if evidence == "runtime" else SimpleNamespace(result_tags=(occurrences, 2))
    traces = [
        {
            "name": tag,
            "values": [False],
            "result_ids": [4] if evidence == "runtime" or (evidence == "mixed" and index == 0) else [],
        }
        for index, (tag, _) in enumerate(layout)
    ]
    if reverse_layout:
        layout = tuple(reversed(layout))
    error = f"generator layout slots {layout[0]!r} and {layout[1]!r} both resolve to measurement id 4"
    with pytest.raises(ValueError, match=re.escape(error)):
        _generator_certified_result_traces(layout, analysis, circuit, traces, required_tags=[])


@pytest.mark.parametrize("runtime_id", [4, 9])
def test_generator_runtime_and_static_must_agree(runtime_id: int) -> None:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [4, 9])
    circuit.set_meta("qis_source_measurement_ids", "[4,9]")
    analysis = SimpleNamespace(result_tags=({"a": [int(runtime_id == 4)], "b": [1]}, 2))
    traces = [
        {"name": "a", "values": [False], "result_ids": [runtime_id]},
        {"name": "b", "values": [False], "result_ids": [9]},
    ]
    other_id = 9 if runtime_id == 4 else 4
    with pytest.raises(
        ValueError,
        match=rf"runtime result trace 'a'\[0\] has measurement id {runtime_id}, "
        rf"but compiled analysis identifies measurement id {other_id}",
    ):
        _generator_certified_result_traces((("a", 0), ("b", 0)), analysis, circuit, traces, required_tags=[])


@pytest.mark.parametrize("case", ["none", "absent", "short", "array", "mixed", "base_array"])
def test_generator_static_evidence_requires_own_scalar_occurrence(case: str) -> None:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [4, 9])
    circuit.set_meta("qis_source_measurement_ids", "[4,9]")
    tag = "physical:meas:0" if case == "base_array" else "physical"
    layout = ((tag, 0), (tag, 1))
    occurrences = {tag: [0, 1]}
    traces = [
        {"name": tag, "values": [False], "result_ids": []},
        {"name": tag, "values": [True], "result_ids": []},
    ]
    if case == "none":
        occurrences[tag][1] = None
    elif case == "absent":
        occurrences = {}
    elif case == "short":
        occurrences[tag] = [0]
    elif case == "array":
        traces = [{"name": tag, "values": [False, True], "result_ids": []}]
    elif case == "mixed":
        # Even an empty non-scalar call invalidates static occurrence matching.
        traces.append({"name": tag, "values": [], "result_ids": []})
    else:
        occurrences = {"physical": [0, 1]}
    index = 1 if case in {"none", "short"} else 0
    with pytest.raises(ValueError, match=rf"{tag!r}\[{index}\]: neither runtime provenance nor compiled analysis"):
        _generator_certified_result_traces(
            layout,
            SimpleNamespace(result_tags=(occurrences, 2)),
            circuit,
            traces,
            required_tags=[],
        )


def test_generator_static_count_mismatch_is_hard_error() -> None:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0], [4])
    circuit.set_meta("qis_source_measurement_ids", "[4]")
    with pytest.raises(_CertificationError, match=r"has 2 measurement\(s\).*emitted 1"):
        _generator_certified_result_traces(
            (("a", 0),),
            SimpleNamespace(result_tags=({"a": [0]}, 2)),
            circuit,
            [{"name": "a", "values": [False], "result_ids": [4]}],
            required_tags=[],
        )


def test_generator_unexpected_analysis_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0], [4])
    circuit.set_meta("qis_source_measurement_ids", "[4]")
    error = RuntimeError("analysis failure")
    monkeypatch.setattr(_qis_result_tags, "extract_result_tag_measurements", Mock(side_effect=error))
    analysis = _GuppyProgramAnalysis(b"")
    analysis.qis = "unused"
    with pytest.raises(RuntimeError, match="analysis failure") as caught:
        _generator_certified_result_traces(
            (("a", 0),),
            analysis,
            circuit,
            [{"name": "a", "values": [False], "result_ids": [4]}],
            required_tags=[],
        )
    assert caught.value is error


@pytest.mark.parametrize("ordinal", [-1, 1])
def test_generator_static_ordinal_bounds(ordinal: int) -> None:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0], [4])
    circuit.set_meta("qis_source_measurement_ids", "[4]")
    with pytest.raises(ValueError, match=f"compiler measurement ordinal {ordinal} is outside"):
        _generator_certified_result_traces(
            (("a", 0),),
            SimpleNamespace(result_tags=({"a": [ordinal]}, 1)),
            circuit,
            [{"name": "a", "values": [False], "result_ids": []}],
            required_tags=[],
        )


def test_generator_layout_rejects_source_runtime_identity_mismatch() -> None:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [4, 9])
    circuit.set_meta("guppy_source_measurement_ids", "[0,1]")
    traces = [
        {"name": "first", "values": [False], "result_ids": [4]},
        {"name": "second", "values": [False], "result_ids": [9]},
    ]

    with pytest.raises(ValueError, match="source and runtime measurement identities"):
        _generator_certified_result_traces(
            (("first", 0), ("second", 0)),
            _UnavailableResultAnalysis(),
            circuit,
            traces,
            required_tags=["first"],
        )


def test_generator_layout_resolves_array_elements_from_runtime_ids() -> None:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [4, 9])
    circuit.set_meta("guppy_source_measurement_ids", "[4,9]")

    bound = _generator_certified_result_traces(
        (("physical", 0), ("physical", 1)),
        _UnavailableResultAnalysis(),
        circuit,
        [{"name": "physical", "values": [False, True], "result_ids": [9, 4]}],
        required_tags=["physical"],
    )

    assert [item["result_ids"] for item in bound] == [[9], [4]]


def test_generator_layout_rejects_array_without_evidence() -> None:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [4, 9])
    circuit.set_meta("guppy_source_measurement_ids", "[4,9]")

    with pytest.raises(
        ValueError,
        match=r"'physical'\[0\]: neither runtime provenance nor compiled analysis identifies its measurement",
    ):
        _generator_certified_result_traces(
            (("physical", 0), ("physical", 1)),
            _UnavailableResultAnalysis(),
            circuit,
            [{"name": "physical", "values": [False, True], "result_ids": []}],
            required_tags=["physical"],
        )


def test_generator_layout_resolves_synthetic_array_slots_from_base_array_ids() -> None:
    """Guppy v1 array provenance remains ordered even with synthetic tags."""
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [4, 9])
    circuit.set_meta("guppy_source_measurement_ids", "[4,9]")
    traces = [
        {"name": "physical:meas:0", "values": [False], "result_ids": []},
        {"name": "physical:meas:1", "values": [True], "result_ids": []},
        {"name": "physical", "values": [False, True], "result_ids": [9, 4]},
    ]

    bound = _generator_certified_result_traces(
        (("physical:meas:0", 0), ("physical:meas:1", 0)),
        _UnavailableResultAnalysis(),
        circuit,
        traces,
        required_tags=["physical:meas:0", "physical:meas:1"],
    )

    assert [item["result_ids"] for item in bound] == [[9], [4]]


def test_generated_surface_layout_certificate_rejects_permutation() -> None:
    program = make_surface_code(3, 1, "Z")
    digest, layout = program.__pecos_named_measurement_layout_v2__
    object.__setattr__(
        program,
        "__pecos_named_measurement_layout_v2__",
        (digest, (layout[1], layout[0], *layout[2:])),
    )

    with pytest.raises(ValueError, match="does not match the program and layout"):
        build_dem_from_guppy(
            program,
            num_qubits=get_num_qubits(3),
            detectors=[Detector(result_ref("final:meas:0"))],
            p1=0.0,
            p2=0.0,
            p_meas=0.1,
            p_prep=0.0,
        )


def test_result_columns_are_a_trusted_carrier_bound_only_by_row_index() -> None:
    """Pin the documented boundary: a permuted column silently changes shots.

    ``evaluate_result_columns`` binds shot i to row i of every column; it has
    no cross-column shot identity to check against. This test exists so the
    boundary is deliberate -- if binding is ever added, update the docstring
    and replace this test with a detection assertion.
    """
    circuit, traces = _reordered_trace()
    schema = _resolve_dem_specs(
        [Detector(result_ref("a"), result_ref("b"))],
        [],
        circuit=circuit,
        result_traces=traces,
    )
    build = GuppyDemBuild(
        dem=None,
        circuit=circuit,
        detectors_json=schema.detectors_json,
        observables_json=schema.observables_json,
        measurement_ledger=schema.ledger,
        schema_fingerprint=schema.schema_fingerprint,
        named_result_binding="compiler_direct_scalar_complete",
        _detector_meas_ids=schema.detector_meas_ids,
        _observable_meas_ids=schema.observable_meas_ids,
        _result_ids_by_tag=schema.result_ids_by_tag,
    )
    columns = {"a": [0, 1], "b": [0, 0], "c": [0, 0]}
    permuted = {**columns, "a": [1, 0]}

    baseline = build.evaluate_result_columns(columns)
    swapped = build.evaluate_result_columns(permuted)

    assert baseline == [([0], 0), ([1], 0)]
    assert swapped == [([1], 0), ([0], 0)]
    assert baseline != swapped


def test_schema_fingerprint_binds_runtime_order_and_named_results() -> None:
    circuit, traces = _reordered_trace()
    baseline = _resolve_dem_specs(
        [Detector(result_ref("c"))],
        [],
        circuit=circuit,
        result_traces=traces,
    )

    reordered = TickCircuit()
    reordered.tick().mz_with_ids([0, 1, 2], [0, 1, 2])
    reordered_schema = _resolve_dem_specs(
        [Detector(result_ref("c"))],
        [],
        circuit=reordered,
        result_traces=traces,
    )
    renamed_traces = [*traces[:-1], {"name": "renamed", "values": [False], "result_ids": [2]}]
    renamed_schema = _resolve_dem_specs(
        [Detector(result_ref("renamed"))],
        [],
        circuit=circuit,
        result_traces=renamed_traces,
    )

    assert baseline.schema_fingerprint != reordered_schema.schema_fingerprint
    assert baseline.schema_fingerprint != renamed_schema.schema_fingerprint


@pytest.mark.parametrize("certified", [False, True])
def test_build_compiles_once_and_traces_analyzed_text(monkeypatch: pytest.MonkeyPatch, *, certified: bool) -> None:
    from pecos import tracing
    from pecos_rslib import hugr_lowering

    compiler = Mock(wraps=hugr_lowering.compile_hugr_to_qis)
    wrapper = Mock(wraps=pecos.Qis)
    tracer = Mock(wraps=tracing.trace_program_to_tick_circuit)
    extractor = Mock(wraps=_qis_result_tags.extract_result_tag_measurements)
    monkeypatch.setattr(hugr_lowering, "compile_hugr_to_qis", compiler)
    monkeypatch.setattr(pecos.programs, "Qis", wrapper)
    monkeypatch.setattr(tracing, "trace_program_to_tick_circuit", tracer)
    monkeypatch.setattr(_qis_result_tags, "extract_result_tag_measurements", extractor)
    build_dem_from_guppy(
        make_surface_code(3, 1, "Z") if certified else _scrambled_tagged_measurements,
        num_qubits=get_num_qubits(3) if certified else 2,
        detectors=[Detector(result_ref("final:meas:0" if certified else "a"))],
        p_meas=0.1,
    )
    assert compiler.call_count == tracer.call_count == wrapper.call_count == 1
    traced_program = tracer.call_args.args[0]
    assert isinstance(traced_program, pecos.Qis)
    extractor.assert_called_once_with(wrapper.call_args.args[0])
    assert extractor.call_args.args[0] is wrapper.call_args.args[0]


@pytest.mark.parametrize("references", ["typed", "json", "empty"])
def test_qis_certification_failure_policy(monkeypatch: pytest.MonkeyPatch, references: str) -> None:
    extractor = Mock(side_effect=_CertificationError("test unsupported CFG"))
    monkeypatch.setattr(_qis_result_tags, "extract_result_tag_measurements", extractor)

    def build() -> None:
        if references == "json":
            DetectorErrorModel.from_guppy(
                _scrambled_tagged_measurements,
                num_qubits=2,
                detectors_json='[{"id":0,"result_tags":["a"]}]',
            )
        else:
            build_dem_from_guppy(
                _scrambled_tagged_measurements,
                num_qubits=2,
                detectors=[] if references == "empty" else [Detector(result_ref("a"))],
            )

    if references == "empty":
        build()
    else:
        reference = "result_tags" if references == "json" else "result_ref"
        with pytest.raises(ValueError, match=_certification_message(reference)) as raised:
            build()
        assert isinstance(raised.value.__cause__, _CertificationError)
        assert str(raised.value.__cause__).endswith("test unsupported CFG")
    extractor.assert_called_once()


def test_unrelated_analysis_value_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_qis_result_tags, "extract_result_tag_measurements", Mock(side_effect=ValueError("unrelated")))
    with pytest.raises(ValueError, match=r"^unrelated$"):
        build_dem_from_guppy(_scrambled_tagged_measurements, num_qubits=2, detectors=[Detector(result_ref("a"))])


@pytest.mark.parametrize("ir", ["define void @qmain(", "define void @qmain() {\n%f =\nret void\n}"])
def test_parser_failure_without_required_tags_skips_certification(monkeypatch: pytest.MonkeyPatch, ir: str) -> None:
    extract = _qis_result_tags.extract_result_tag_measurements
    monkeypatch.setattr(_qis_result_tags, "extract_result_tag_measurements", lambda _qis: extract(ir))
    built = build_dem_from_guppy(_scrambled_tagged_measurements, num_qubits=2, detectors=[])
    assert built.named_result_binding == "none"


@pytest.mark.parametrize("required", [False, True])
def test_certified_measurement_count_mismatch(monkeypatch: pytest.MonkeyPatch, *, required: bool) -> None:
    monkeypatch.setattr(
        _qis_result_tags,
        "extract_result_tag_measurements",
        Mock(return_value=({"a": [0], "b": [1]}, 3)),
    )

    def build() -> None:
        build_dem_from_guppy(
            _scrambled_tagged_measurements,
            num_qubits=2,
            detectors=[Detector(result_ref("a"))] if required else [],
        )

    if required:
        with pytest.raises(
            ValueError,
            match=_certification_message("result_ref"),
        ) as raised:
            build()
        assert isinstance(raised.value.__cause__, _CertificationError)
        assert str(raised.value.__cause__).endswith(
            "the certified lowered program has 3 measurement(s) but the traced program emitted 2 measurement(s)",
        )
    else:
        build()


def _certification_message(reference: str) -> str:
    return (
        "^"
        + re.escape(
            f"Cannot bind {reference}: the program's lowered form could not be certified. "
            "For a measurement-independent quantum schedule with raw and parity outputs, use "
            "infer_guppy_dem_annotations(program, num_qubits=...).build_dem(). Alternatively, pass "
            "DetectorErrorModel.from_guppy(..., detectors_json=..., observables_json=...) JSON with "
            "meas_ids and no result_tags; the static-schedule preflight still applies. MeasIds label "
            "compiled measurements, not source positions: take them from "
            "infer_guppy_dem_annotations(...).detectors_json / .observables_json or from a traced "
            "GuppyDemBuild.measurement_ledger entry's .meas_id.",
        )
        + "$"
    )


def test_json_measurement_count_guard_stays_in_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        _qis_result_tags,
        "extract_result_tag_measurements",
        Mock(return_value=({"a": [0], "b": [1]}, 3)),
    )
    with pytest.raises(
        ValueError,
        match=(
            r"^DEM builder parse error: Cannot bind result_tags: "
            r"the certified lowered program has 3 measurement\(s\) but "
            r"the traced program emitted 2 measurement\(s\)\.$"
        ),
    ) as raised:
        DetectorErrorModel.from_guppy(
            _scrambled_tagged_measurements,
            num_qubits=2,
            detectors_json='[{"id":0,"result_tags":["a"]}]',
        )
    assert raised.value.__cause__ is None


@guppy
def _measurements_with_partially_traced_integer_tag() -> None:
    q0 = qubit()
    q1 = qubit()
    result("m", measure(q0).read())
    result("n", 1)
    result("n", 3)
    result("k", measure(q1).read())


@pytest.mark.parametrize("required", [True, False], ids=["bool-tag", "integer-tag"])
def test_integer_prints_preserve_measurement_bindings(*, required: bool) -> None:
    built = build_dem_from_guppy(
        _measurements_with_partially_traced_integer_tag,
        num_qubits=2,
        detectors=[Detector(result_ref("m") if required else result_ref("k"))],
        p_meas=0.1,
    )
    assert built.named_result_binding == "compiler_direct_scalar_partial"
    assert built.evaluate_results({"m": 1, "k": 1}) == ([1], 0)
    # Distinct values prove that m binds q0, and the second named reference binds q1.
    assert built.evaluate_results({"m": 1, "k": 0}) == ([int(required)], 0)


def test_integer_prints_preserve_json_measurement_binding() -> None:
    _, offsets = measurement_layout(_measurements_with_partially_traced_integer_tag, 2)

    def dem(detectors_json: str) -> str:
        return DetectorErrorModel.from_guppy(
            _measurements_with_partially_traced_integer_tag,
            num_qubits=2,
            detectors_json=detectors_json,
            p_meas=0.1,
        ).to_string()

    assert dem('[{"id":0,"result_tags":["m"]}]') == dem(json.dumps([{"id": 0, "records": [offsets[0]]}]))


@pytest.mark.parametrize("references", ["typed", "json"])
def test_integer_prints_keep_nonmeasurement_diagnostic(references: str) -> None:
    if references == "typed":
        with pytest.raises(
            ValueError,
            match=r"^result_ref 'n' occurrence 0 is not a direct scalar measurement result$",
        ):
            build_dem_from_guppy(
                _measurements_with_partially_traced_integer_tag,
                num_qubits=2,
                detectors=[Detector(result_ref("n"))],
            )
    else:
        with pytest.raises(
            ValueError,
            match=(
                r'^DEM builder parse error: Detector references result_tag "n", '
                r"but occurrence 0 is not a direct scalar measurement result$"
            ),
        ):
            DetectorErrorModel.from_guppy(
                _measurements_with_partially_traced_integer_tag,
                num_qubits=2,
                detectors_json='[{"id":0,"result_tags":["n"]}]',
            )


@pytest.mark.parametrize("invalid_id", [True, -1, 99, "4", None])
def test_generator_evidence_requires_a_traced_measurement_id(invalid_id):
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0], [4])
    circuit.set_meta("qis_source_measurement_ids", "[4]")
    with pytest.raises(ValueError, match="invalid measurement id"):
        _generator_certified_result_traces(
            (("out", 0),),
            _UnavailableResultAnalysis(),
            circuit,
            [{"name": "out", "values": [False], "result_ids": [invalid_id]}],
            required_tags=[],
        )


@pytest.mark.parametrize("role", ["detector", "observable"])
@pytest.mark.parametrize("entrypoint", ["wrapper", "builder"])
@pytest.mark.parametrize("mixed", [False, True])
def test_typed_guppy_rejects_positional_references(
    role: str,
    entrypoint: str,
    mixed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_execution(*_args, **_kwargs):
        pytest.fail("record references must be rejected before compilation, tracing or inference")

    for target in (
        "pecos.qec.dem._preflight_guppy_static_schedule",
        "pecos.tracing._trace_program_to_tick_circuit_with_result_traces",
        "pecos.qec.guppy_output_dem.infer_guppy_dem_annotations",
    ):
        monkeypatch.setattr(target, unexpected_execution)
    detectors = [Detector(rec[-1], id=0)] if role == "detector" else []
    observables = [Observable(rec[-1], id=7)] if role == "observable" else []
    if mixed:
        if role == "detector":
            detectors.append(Detector.inferred("checks", id=1))
        else:
            observables.append(Observable.inferred("logical", id=8))
    identity = 0 if role == "detector" else 7
    expected = (
        f"{role} {identity} uses rec[-1]: compiled Guppy programs do not preserve source measurement order; "
        "use result_ref(tag), or result_ref(tag, occurrence=...) for repeated tags"
    )

    def build():
        if entrypoint == "wrapper":
            build_dem_from_guppy(
                _scrambled_tagged_measurements,
                num_qubits=2,
                detectors=detectors,
                observables=observables,
            )
        else:
            (
                DetectorErrorModel.builder()
                .with_program(_scrambled_tagged_measurements)
                .with_qubits(2)
                .with_detectors(detectors)
                .with_observables(observables)
                .build()
            )

    with pytest.raises(TypeError, match=re.escape(expected)):
        build()
