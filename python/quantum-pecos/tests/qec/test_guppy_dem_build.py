# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Adversarial measurement-identity tests for typed Guppy DEM builds."""

from __future__ import annotations

import json

import pecos
import pytest
from guppylang import guppy
from guppylang.std.builtins import array, result
from guppylang.std.quantum import cx, h, measure, qubit, x
from pecos._qis_trace_replay import (
    _replay_qis_trace_chunks_into_tick_circuit,
    _validate_audited_trace_stream,
)
from pecos.guppy_gen import get_num_qubits, make_surface_code
from pecos.qec import (
    Detector,
    Observable,
    build_dem_from_guppy,
    rec,
    result_ref,
    surface_memory_dem_spec,
)
from pecos.qec.dem import _generator_certified_result_traces
from pecos.qec.dem_spec import GuppyDemBuild, _resolve_dem_specs
from pecos_rslib.quantum import TickCircuit


@guppy
def _scrambled_tagged_measurements() -> None:
    qa = qubit()
    qb = qubit()
    h(qb)
    cx(qb, qa)
    a = measure(qa)
    b = measure(qb)
    result("b", b)
    result("a", a)


@guppy
def _transformed_measurement_result() -> None:
    q = qubit()
    measured = measure(q)
    result("not_m", not measured)


@guppy
def _aggregate_measurement_result() -> None:
    qa = qubit()
    qb = qubit()
    a = measure(qa)
    b = measure(qb)
    result("pair", array(a, b))


@guppy
def _mixed_supported_result_occurrences() -> None:
    q = qubit()
    measured = measure(q)
    result("same", not measured)
    result("same", measured)


@guppy
def _measurement_feedback_without_named_results() -> None:
    q0 = qubit()
    q1 = qubit()
    h(q0)
    if measure(q0):
        x(q1)
    _ = measure(q1)


def _reordered_trace() -> tuple[TickCircuit, list[dict[str, object]]]:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([2, 0, 1], [2, 0, 1])
    traces = [
        {"name": "a", "values": [False], "result_ids": [0]},
        {"name": "b", "values": [False], "result_ids": [1]},
        {"name": "c", "values": [False], "result_ids": [2]},
    ]
    return circuit, traces


def test_rec_and_result_refs_resolve_identically_after_runtime_reorder() -> None:
    circuit, traces = _reordered_trace()
    via_records = _resolve_dem_specs(
        [Detector(rec[-3], rec[-1])],
        [Observable(rec[-2])],
        circuit=circuit,
        result_traces=traces,
    )
    via_results = _resolve_dem_specs(
        [Detector(result_ref("a"), result_ref("c"))],
        [Observable(result_ref("b"))],
        circuit=circuit,
        result_traces=traces,
    )

    assert via_records.detectors_json == via_results.detectors_json
    assert via_records.observables_json == via_results.observables_json
    assert via_records.schema_fingerprint == via_results.schema_fingerprint
    assert json.loads(via_records.detectors_json) == [{"id": 0, "meas_ids": [0, 2]}]
    assert [entry.meas_id for entry in via_records.ledger] == [2, 0, 1]


def test_real_guppy_rec_and_result_ref_builds_are_byte_identical() -> None:
    noise = {"p1": 0.01, "p2": 0.02, "p_meas": 0.1, "p_prep": 0.0}
    via_records = build_dem_from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=2,
        detectors=[Detector(rec[-2])],
        observables=[Observable(rec[-1])],
        **noise,
    )
    via_results = build_dem_from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=2,
        detectors=[Detector(result_ref("a"))],
        observables=[Observable(result_ref("b"))],
        **noise,
    )

    assert via_records.detectors_json == via_results.detectors_json
    assert via_records.observables_json == via_results.observables_json
    assert via_records.schema_fingerprint == via_results.schema_fingerprint
    assert via_records.dem.to_string() == via_results.dem.to_string()


def test_trace_once_build_evaluates_runtime_and_rejects_uncertified_named_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_trace(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return _reordered_trace()

    monkeypatch.setattr(
        "pecos.tracing._trace_program_to_tick_circuit_with_result_traces",
        fake_trace,
    )
    build = build_dem_from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=3,
        detectors=[
            Detector(rec[-3], rec[-1]),
            Detector(rec[-3], rec[-1]),
        ],
        observables=[Observable(rec[-2])],
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
        build.evaluate_results({"a": [0], "b": [1], "c": [1]})


def test_rec_build_evaluates_compiler_certified_scalar_results() -> None:
    build = build_dem_from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=2,
        detectors=[Detector(rec[-2])],
        observables=[Observable(rec[-1])],
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
    )

    assert build.audit["named_result_binding"] == "compiler_direct_scalar_complete"
    assert build.evaluate_results({"a": [0], "b": [1]}) == ([0], 1)


def test_typed_rec_build_rejects_dynamic_control_without_named_results() -> None:
    with pytest.raises(ValueError, match="branching or looping control flow"):
        build_dem_from_guppy(
            _measurement_feedback_without_named_results,
            num_qubits=2,
            detectors=[Detector(rec[-2], rec[-1])],
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

    with pytest.raises(ValueError, match="HUGR-certifiable"):
        build_dem_from_guppy(
            object(),
            num_qubits=1,
            detectors=[Detector(rec[-1])],
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
                detectors=[Detector(rec[-2], rec[-1])],
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
            detectors=[Detector(rec[-2])],
            observables=[Observable(rec[-1])],
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


def test_forged_certificate_on_byte_carrier_does_not_bypass_the_guard() -> None:
    """A self-consistent digest stapled to a pecos.Hugr wrapper must not
    suppress the control-flow guard: certificates are honored only on Guppy
    definition objects, never on byte carriers."""
    import hashlib as _hashlib

    import pecos
    from pecos._compilation import guppy_to_hugr

    dynamic_bytes = guppy_to_hugr(_measurement_feedback_without_named_results)
    layout_json = json.dumps([], separators=(",", ":"))
    digest = _hashlib.sha256(dynamic_bytes + b"\0" + layout_json.encode()).hexdigest()
    forged = pecos.Hugr(dynamic_bytes)
    forged.__pecos_named_measurement_layout_v2__ = (digest, [])

    with pytest.raises(ValueError, match="branching or looping control flow"):
        build_dem_from_guppy(
            forged,
            num_qubits=2,
            detectors=[Detector(rec[-2], rec[-1])],
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
            [Detector(rec[-1], metadata={"bad": {1, 2}})],
            [],
            circuit=circuit,
            result_traces=traces,
        )
    with pytest.raises(ValueError, match="JSON-serializable"):
        _resolve_dem_specs(
            [Detector(rec[-1], metadata={"bad": float("nan")})],
            [],
            circuit=circuit,
            result_traces=traces,
        )
    with pytest.raises(ValueError, match="even multiplicity"):
        _resolve_dem_specs(
            [Detector(rec[-1], rec[-1])],
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
            detectors=[Detector(rec[-2], rec[-1])],
            p1=0.0,
            p2=0.0,
            p_meas=0.1,
            p_prep=0.0,
        )


def test_rec_build_rejects_transformed_scalar_result_as_measurement_provenance() -> None:
    build = build_dem_from_guppy(
        _transformed_measurement_result,
        num_qubits=1,
        detectors=[Detector(rec[-1])],
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
    )

    assert build.audit["named_result_binding"] == "none"
    with pytest.raises(ValueError, match="missing referenced MeasId"):
        build.evaluate_results({"not_m": [1]})


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


def test_rec_build_rejects_aggregate_array_as_measurement_provenance() -> None:
    build = build_dem_from_guppy(
        _aggregate_measurement_result,
        num_qubits=2,
        detectors=[Detector(rec[-2])],
        p1=0.0,
        p2=0.0,
        p_meas=0.1,
        p_prep=0.0,
    )

    with pytest.raises(ValueError, match="missing referenced MeasId"):
        build.evaluate_results({"pair": [0, 1]})


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
        [Detector(rec[-3], rec[-1])],
        [],
        circuit=circuit,
        result_traces=traces,
    )

    assert dict(schema.result_ids_by_tag) == {"a": (0,), "b": (1,), "final": (2,)}


def test_rec_rejects_non_dense_runtime_measurement_identities() -> None:
    circuit = TickCircuit()
    circuit.tick().mz_with_ids([0, 1], [4, 9])

    with pytest.raises(ValueError, match=r"rec\[\.\.\.\].*dense"):
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
    build_dem_from_guppy(
        _scrambled_tagged_measurements,
        num_qubits=3,
        detectors=[Detector(rec[-1])],
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
            circuit,
            traces,
            required_tags=["first"],
        )


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
            detectors=[Detector(rec[-1])],
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
        [Detector(rec[-1])],
        [],
        circuit=circuit,
        result_traces=traces,
    )

    reordered = TickCircuit()
    reordered.tick().mz_with_ids([0, 1, 2], [0, 1, 2])
    reordered_schema = _resolve_dem_specs(
        [Detector(rec[-1])],
        [],
        circuit=reordered,
        result_traces=traces,
    )
    renamed_traces = [*traces[:-1], {"name": "renamed", "values": [False], "result_ids": [2]}]
    renamed_schema = _resolve_dem_specs(
        [Detector(rec[-1])],
        [],
        circuit=circuit,
        result_traces=renamed_traces,
    )

    assert baseline.schema_fingerprint != reordered_schema.schema_fingerprint
    assert baseline.schema_fingerprint != renamed_schema.schema_fingerprint
