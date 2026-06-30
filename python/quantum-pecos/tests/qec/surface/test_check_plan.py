# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Surface check-plan metadata tests."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import replace

import pytest

_PACKED_TRACE_METADATA_JSON_KEY = "__pecos_trace_metadata_json_v1__"


def _packed_trace_metadata_records(source: str) -> list[tuple[int, dict[str, str]]]:
    records: list[tuple[int, dict[str, str]]] = []
    sentinel = f"{json.dumps(_PACKED_TRACE_METADATA_JSON_KEY)}, "
    for line_index, line in enumerate(source.splitlines()):
        if "pecos_qis_trace_metadata_qubit_hugr(" not in line or sentinel not in line:
            continue
        packed_literal = line.split(sentinel, 1)[1].rsplit(")", 1)[0]
        records.append((line_index, json.loads(ast.literal_eval(packed_literal))))
    return records


def _first_metadata_line(
    source: str,
    *,
    start_line: int = 0,
    **expected: str,
) -> tuple[int, dict[str, str]]:
    for line_index, metadata in _packed_trace_metadata_records(source):
        if line_index < start_line:
            continue
        if all(metadata.get(key) == value for key, value in expected.items()):
            return line_index, metadata
    msg = f"could not find packed trace metadata after line {start_line}: {expected!r}"
    raise AssertionError(msg)


def _has_metadata_prefix(source: str, key: str, prefix: str) -> bool:
    return any(
        isinstance(metadata.get(key), str) and metadata[key].startswith(prefix)
        for _, metadata in _packed_trace_metadata_records(source)
    )


def test_check_plan_default_resolves_to_cx_metadata() -> None:
    from pecos.qec.surface._check_plan import (
        canonical_check_plan_json,
        default_surface_check_plan_id,
        resolve_surface_check_plan,
        surface_check_plan_ids,
    )

    plan = resolve_surface_check_plan()

    assert surface_check_plan_ids() == (
        "cx_balanced_data_v1",
        "cx_standard_v1",
        "szz_balanced_data_round_order_3102_v1",
        "szz_balanced_data_v1",
        "szz_boundary_first_balanced_data_v1",
        "szz_boundary_first_v1",
        "szz_current_v1",
    )
    assert default_surface_check_plan_id() == "cx_standard_v1"
    assert default_surface_check_plan_id("SZZ") == "szz_current_v1"
    assert plan.plan_id == "cx_standard_v1"
    assert plan.interaction_basis == "cx"
    assert plan.synthesis_identity == {
        "family": "cx",
        "szz_phase_pattern": "none",
        "interaction_order": "pecos-default",
        "ancilla_schedule": "default",
    }
    assert plan.resolved_metadata["metadata_version"] == 1
    assert plan.resolved_metadata["hash_algorithm"] == "sha256"
    assert plan.resolved_metadata["hash_serialization"] == "canonical-json-v1"
    assert "metadata_version" not in plan.semantic_content
    assert plan.resolved_hash == hashlib.sha256(
        canonical_check_plan_json(plan.semantic_content).encode("utf-8"),
    ).hexdigest()


def test_check_plan_is_source_of_truth_for_basis() -> None:
    from pecos.qec.surface._check_plan import resolve_surface_check_plan

    plan = resolve_surface_check_plan(check_plan="szz_current_v1")

    assert plan.plan_id == "szz_current_v1"
    assert plan.interaction_basis == "szz"
    assert plan.synthesis_identity == {
        "family": "szz",
        "szz_phase_pattern": "standard",
        "interaction_order": "pecos-default",
        "ancilla_schedule": "default",
    }


def test_boundary_first_szz_check_plan_resolves_to_concrete_synthesis() -> None:
    from pecos.qec.surface._check_plan import resolve_surface_check_plan

    plan = resolve_surface_check_plan(check_plan="szz_boundary_first_v1")

    assert plan.plan_id == "szz_boundary_first_v1"
    assert plan.interaction_basis == "szz"
    assert plan.synthesis_identity == {
        "family": "szz",
        "szz_phase_pattern": "boundary-first",
        "interaction_order": "pecos-default",
        "ancilla_schedule": "default",
    }
    assert plan.semantic_content["x_check"]["sign_policy"] == "boundary_first_szz_sign_vector_v1"
    assert plan.semantic_content["z_check"]["sign_policy"] == "boundary_first_szz_sign_vector_v1"


@pytest.mark.parametrize(
    ("plan_id", "interaction_basis", "phase_pattern"),
    [
        ("cx_balanced_data_v1", "cx", "none"),
        ("szz_balanced_data_v1", "szz", "standard"),
        ("szz_boundary_first_balanced_data_v1", "szz", "boundary-first"),
    ],
)
def test_balanced_data_check_plans_resolve_to_explicit_schedule(
    plan_id: str,
    interaction_basis: str,
    phase_pattern: str,
) -> None:
    from pecos.qec.surface._check_plan import (
        ancilla_schedule_for_check_plan,
        require_current_surface_check_plan_renderer,
        resolve_surface_check_plan,
    )

    plan = resolve_surface_check_plan(check_plan=plan_id)

    assert plan.interaction_basis == interaction_basis
    assert plan.synthesis_identity == {
        "family": interaction_basis,
        "szz_phase_pattern": phase_pattern,
        "interaction_order": "pecos-default",
        "ancilla_schedule": "balanced-data-v1",
    }
    assert plan.semantic_content["schedule"]["ancilla_batch_policy"] == "balanced-data-v1"
    assert ancilla_schedule_for_check_plan(plan) == "balanced-data-v1"
    require_current_surface_check_plan_renderer(plan, context="unit-test")


def test_round_order_check_plan_resolves_to_explicit_schedule() -> None:
    from pecos.qec.surface._check_plan import (
        cnot_round_order_for_check_plan,
        require_current_surface_check_plan_renderer,
        resolve_surface_check_plan,
    )
    from pecos.qec.surface.schedule import CNOT_ROUND_ORDER_3102

    plan = resolve_surface_check_plan(check_plan="szz_balanced_data_round_order_3102_v1")

    assert plan.interaction_basis == "szz"
    assert plan.synthesis_identity == {
        "family": "szz",
        "szz_phase_pattern": "standard",
        "interaction_order": "pecos-default",
        "ancilla_schedule": "balanced-data-v1",
    }
    assert plan.semantic_content["schedule"] == {
        "round_policy": "constant",
        "site_policy": "global",
        "edge_order": "current_surface_cnot_schedule_v1",
        "ancilla_batch_policy": "balanced-data-v1",
        "round_order": CNOT_ROUND_ORDER_3102,
    }
    assert cnot_round_order_for_check_plan(plan) == CNOT_ROUND_ORDER_3102
    require_current_surface_check_plan_renderer(plan, context="unit-test")


def test_current_renderer_rejects_unimplemented_plan_semantics() -> None:
    from pecos.qec.surface._check_plan import require_current_surface_check_plan_renderer, resolve_surface_check_plan

    plan = resolve_surface_check_plan(check_plan="szz_current_v1")
    unsupported = dict(plan.semantic_content)
    unsupported["synthesis_identity"] = {
        **dict(plan.synthesis_identity),
        "szz_phase_pattern": "checkerboard",
    }
    unsupported_plan = replace(
        plan,
        synthesis_identity=dict(unsupported["synthesis_identity"]),
        semantic_content=unsupported,
    )

    with pytest.raises(NotImplementedError, match=r"unit-test.*checkerboard"):
        require_current_surface_check_plan_renderer(
            unsupported_plan,
            context="unit-test",
        )


def test_check_plan_and_interaction_basis_mismatch_fails_loudly() -> None:
    from pecos.qec.surface._check_plan import resolve_surface_check_plan

    with pytest.raises(ValueError, match="conflicts with check_plan"):
        resolve_surface_check_plan(
            interaction_basis="cx",
            check_plan="szz_current_v1",
        )


def test_guppy_surface_code_module_records_resolved_check_plan() -> None:
    from pecos.guppy import get_surface_code_module

    module = get_surface_code_module(3, check_plan="szz_current_v1")

    assert module["interaction_basis"] == "szz"
    assert module["check_plan"] == "szz_current_v1"
    assert module["resolved_check_plan"]["semantic_content"]["interaction_basis"] == "szz"
    assert len(module["resolved_check_plan_hash"]) == 64


def test_guppy_surface_code_rejects_plan_basis_mismatch() -> None:
    from pecos.guppy import make_surface_code

    with pytest.raises(ValueError, match="conflicts with check_plan"):
        make_surface_code(
            distance=3,
            num_rounds=1,
            basis="Z",
            interaction_basis="cx",
            check_plan="szz_current_v1",
        )


def test_guppy_surface_code_accepts_check_plan_as_source_of_truth() -> None:
    from pecos.guppy import make_surface_code

    program = make_surface_code(
        distance=3,
        num_rounds=1,
        basis="Z",
        check_plan="szz_current_v1",
    )

    assert program is not None


@pytest.mark.parametrize(
    "check_plan",
    [
        "cx_balanced_data_v1",
        "szz_balanced_data_v1",
        "szz_balanced_data_round_order_3102_v1",
        "szz_boundary_first_balanced_data_v1",
    ],
)
def test_guppy_surface_code_accepts_balanced_data_check_plans(check_plan: str) -> None:
    from pecos.guppy import make_surface_code

    program = make_surface_code(
        distance=3,
        num_rounds=1,
        basis="Z",
        ancilla_budget=2,
        check_plan=check_plan,
    )

    assert program is not None


def test_surface_code_memory_records_resolved_check_plan() -> None:
    from pecos.qec.surface import surface_code_memory

    result = surface_code_memory(
        distance=3,
        physical_error_rate=0.0,
        shots=4,
        rounds=1,
        seed=123,
        check_plan="szz_current_v1",
    )

    assert result.interaction_basis == "szz"
    assert result.check_plan == "szz_current_v1"
    assert result.resolved_check_plan is not None
    assert result.resolved_check_plan["semantic_content"]["interaction_basis"] == "szz"
    assert len(result.resolved_check_plan_hash) == 64


def test_surface_code_memory_rejects_plan_basis_mismatch() -> None:
    from pecos.qec.surface import surface_code_memory

    with pytest.raises(ValueError, match="conflicts with check_plan"):
        surface_code_memory(
            distance=3,
            physical_error_rate=0.0,
            shots=0,
            rounds=1,
            interaction_basis="cx",
            check_plan="szz_current_v1",
        )


def test_check_plan_does_not_change_current_szz_dem() -> None:
    from pecos.qec.surface import NoiseModel, SurfacePatch
    from pecos.qec.surface.decode import generate_circuit_level_dem_from_builder

    patch = SurfacePatch.create(distance=3)
    noise = NoiseModel(p2=0.001, p_meas=0.001, p_prep=0.001)

    by_basis = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=1,
        noise=noise,
        interaction_basis="szz",
    )
    by_plan = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=1,
        noise=noise,
        check_plan="szz_current_v1",
    )

    assert by_plan == by_basis


def test_direct_surface_renderers_accept_check_plan_as_source_of_truth() -> None:
    from pecos.qec.surface import SurfacePatch
    from pecos.qec.surface.circuit_builder import (
        generate_dag_circuit_from_patch,
        generate_guppy_from_patch,
        generate_stim_from_patch,
        generate_tick_circuit_from_patch,
    )
    from pecos.qec.surface.decode import build_memory_circuit

    patch = SurfacePatch.create(distance=3)

    stim_text = generate_stim_from_patch(
        patch,
        num_rounds=1,
        check_plan="szz_current_v1",
        add_detectors=False,
    )
    assert "CX" not in stim_text
    assert "SQRT_ZZ" in stim_text

    dag_circuit = generate_dag_circuit_from_patch(
        patch,
        num_rounds=1,
        check_plan="szz_current_v1",
    )
    assert "SZZ" in {dag_circuit.gate(node).gate_type.name for node in dag_circuit.nodes()}

    tick_circuit = generate_tick_circuit_from_patch(
        patch,
        num_rounds=1,
        check_plan="szz_current_v1",
    )
    assert int(tick_circuit.get_meta("num_detectors")) > 0

    memory_circuit = build_memory_circuit(
        patch=patch,
        rounds=1,
        check_plan="szz_current_v1",
    )
    assert int(memory_circuit.get_meta("num_detectors")) == int(tick_circuit.get_meta("num_detectors"))

    guppy_source = generate_guppy_from_patch(patch, check_plan="szz_current_v1")
    assert "Check plan: szz_current_v1" in guppy_source
    assert "def pecos_qis_trace_metadata_qubit_hugr(" in guppy_source
    assert " = pecos_qis_trace_metadata_qubit_hugr(" in guppy_source
    assert _PACKED_TRACE_METADATA_JSON_KEY in guppy_source
    records = [metadata for _, metadata in _packed_trace_metadata_records(guppy_source)]
    assert any(metadata.get("source_kind") == "szz_host" for metadata in records)
    assert any(metadata.get("source_kind") == "szz_data_prefix" for metadata in records)
    assert any(str(metadata.get("szz_host_label", "")).startswith("szz:") for metadata in records)
    assert any(str(metadata.get("host_id", "")).startswith("szz:") for metadata in records)
    assert any(metadata.get("local_role") == "basis_prefix" for metadata in records)
    assert any(metadata.get("source_lowering_required") == "true" for metadata in records)


def test_szz_guppy_source_can_disable_trace_metadata_for_execution() -> None:
    from pecos.guppy.surface import generate_guppy_source
    from pecos.qec.surface import SurfacePatch

    patch = SurfacePatch.create(distance=3)
    guppy_source = generate_guppy_source(
        patch,
        num_rounds=1,
        interaction_basis="szz",
        check_plan="szz_current_v1",
        trace_metadata=False,
    )

    assert "def pecos_qis_trace_metadata_qubit_hugr(" not in guppy_source
    assert "pecos_qis_trace_metadata_qubit_hugr(" not in guppy_source
    assert "zz_phase(" in guppy_source
    assert "result(" in guppy_source


def test_szz_runtime_barrier_fences_data_prefix_before_host() -> None:
    from pecos.guppy.surface import generate_guppy_source
    from pecos.qec.surface import SurfacePatch

    patch = SurfacePatch.create(distance=3)
    source = generate_guppy_source(
        patch,
        check_plan="szz_current_v1",
        szz_runtime_barriers="data-prefix",
    )

    lines = source.splitlines()
    barrier_index = next(i for i, line in enumerate(lines) if "= pecos_qis_runtime_barrier_qubits2_hugr(" in line)
    prefix_index, _ = _first_metadata_line(
        source,
        start_line=barrier_index,
        source_kind="szz_data_prefix",
    )
    host_index, _ = _first_metadata_line(source, start_line=prefix_index, source_kind="szz_host")
    zz_phase_index = next(i for i in range(barrier_index, len(lines)) if "zz_phase(" in lines[i])

    assert barrier_index < prefix_index < host_index < zz_phase_index


def test_szz_data_prefixes_emit_generic_hosted_metadata() -> None:
    from pecos.guppy.surface import generate_guppy_source
    from pecos.qec.surface import SurfacePatch

    source = generate_guppy_source(
        SurfacePatch.create(distance=3),
        check_plan="szz_current_v1",
    )

    prefix_index, prefix_metadata = _first_metadata_line(
        source,
        source_kind="szz_data_prefix",
    )
    host_index, host_metadata = _first_metadata_line(
        source,
        start_line=prefix_index + 1,
        source_kind="szz_host",
    )

    assert prefix_metadata["local_role"] == "basis_prefix"
    assert prefix_metadata["host_id"].startswith("szz:")
    assert host_metadata["host_id"].startswith("szz:")
    assert prefix_index < host_index


def test_szz_hosted_metadata_labels_include_helper_scope() -> None:
    from pecos.guppy.surface import generate_guppy_source
    from pecos.qec.surface import SurfacePatch

    source = generate_guppy_source(
        SurfacePatch.create(distance=3),
        check_plan="szz_current_v1",
    )

    assert _has_metadata_prefix(source, "host_id", "szz:init_z_basis:")
    assert _has_metadata_prefix(source, "host_id", "szz:init_x_basis:")
    assert _has_metadata_prefix(source, "host_id", "szz:syndrome_extraction:")


def test_plain_szz_memory_source_unrolls_hosted_metadata_by_counted_round() -> None:
    from pecos.guppy.surface import generate_guppy_source
    from pecos.qec.surface import SurfacePatch

    source = generate_guppy_source(
        SurfacePatch.create(distance=3),
        check_plan="szz_current_v1",
        num_rounds=2,
    )

    assert "for _t in range(comptime(num_rounds))" not in source
    assert "if num_rounds != 2:" in source
    assert "def syndrome_extraction_memory_r0" in source
    assert "def syndrome_extraction_memory_r1" in source
    assert _has_metadata_prefix(source, "host_id", "szz:memory_r0:")
    assert _has_metadata_prefix(source, "host_id", "szz:memory_r1:")
    assert not _has_metadata_prefix(source, "host_id", "szz:memory_r2:")


def test_plain_szz_memory_cache_key_includes_counted_rounds() -> None:
    from pecos.guppy.surface import _guppy_module_cache_key
    from pecos.qec.surface import SurfacePatch

    patch = SurfacePatch.create(distance=3)

    key_one = _guppy_module_cache_key(
        patch,
        8,
        check_plan="szz_current_v1",
        num_rounds=1,
    )
    key_two = _guppy_module_cache_key(
        patch,
        8,
        check_plan="szz_current_v1",
        num_rounds=2,
    )
    key_generic = _guppy_module_cache_key(
        patch,
        8,
        check_plan="szz_current_v1",
    )

    assert key_one.endswith("_r1")
    assert key_two.endswith("_r2")
    assert key_one != key_two
    assert key_generic not in {key_one, key_two}


def test_boundary_first_szz_check_plan_changes_source_gates_not_metadata() -> None:
    from pecos.qec.surface import SurfacePatch
    from pecos.qec.surface.circuit_builder import (
        OpType,
        build_surface_code_circuit,
        generate_guppy_from_patch,
        generate_tick_circuit_from_patch,
    )

    patch = SurfacePatch.create(distance=3)

    current_ops, _ = build_surface_code_circuit(
        patch,
        num_rounds=1,
        check_plan="szz_current_v1",
    )
    boundary_first_ops, _ = build_surface_code_circuit(
        patch,
        num_rounds=1,
        check_plan="szz_boundary_first_v1",
    )

    def szz_gate_signature(ops: object) -> list[tuple[str, tuple[int, ...], str]]:
        return [
            (op.op_type.name, tuple(op.qubits), op.label)
            for op in ops
            if op.op_type in {OpType.SZZ, OpType.SZZDG}
        ]

    assert szz_gate_signature(boundary_first_ops) != szz_gate_signature(current_ops)
    assert sum(op.op_type == OpType.SZZDG for op in boundary_first_ops) == sum(
        op.op_type == OpType.SZZDG for op in current_ops
    )

    current_tick = generate_tick_circuit_from_patch(
        patch,
        num_rounds=1,
        check_plan="szz_current_v1",
    )
    boundary_first_tick = generate_tick_circuit_from_patch(
        patch,
        num_rounds=1,
        check_plan="szz_boundary_first_v1",
    )
    assert boundary_first_tick.get_meta("detectors") == current_tick.get_meta("detectors")
    assert boundary_first_tick.get_meta("observables") == current_tick.get_meta("observables")

    source = generate_guppy_from_patch(patch, check_plan="szz_boundary_first_v1")
    assert "Check plan: szz_boundary_first_v1" in source


def test_round_order_szz_check_plan_changes_host_order_not_metadata() -> None:
    from pecos.guppy.surface import generate_guppy_source
    from pecos.qec.surface import SurfacePatch
    from pecos.qec.surface.circuit_builder import (
        OpType,
        build_surface_code_circuit,
        generate_tick_circuit_from_patch,
    )

    patch = SurfacePatch.create(distance=3)
    baseline_plan = "szz_balanced_data_v1"
    round_order_plan = "szz_balanced_data_round_order_3102_v1"

    def szz_gate_signature(plan_id: str) -> list[tuple[str, tuple[int, ...], str]]:
        ops, _ = build_surface_code_circuit(
            patch,
            num_rounds=1,
            ancilla_budget=2,
            check_plan=plan_id,
        )
        return [
            (op.label, tuple(op.qubits), op.op_type.name)
            for op in ops
            if op.op_type in {OpType.SZZ, OpType.SZZDG}
        ]

    baseline_signature = szz_gate_signature(baseline_plan)
    round_order_signature = szz_gate_signature(round_order_plan)
    assert round_order_signature != baseline_signature
    assert round_order_signature[:4] == [
        ("X0", (9, 0), "SZZ"),
        ("X0", (9, 1), "SZZDG"),
        ("X3", (9, 7), "SZZ"),
        ("X3", (9, 8), "SZZDG"),
    ]

    baseline_tick = generate_tick_circuit_from_patch(
        patch,
        num_rounds=1,
        ancilla_budget=2,
        check_plan=baseline_plan,
    )
    round_order_tick = generate_tick_circuit_from_patch(
        patch,
        num_rounds=1,
        ancilla_budget=2,
        check_plan=round_order_plan,
    )
    assert round_order_tick.get_meta("detectors") == baseline_tick.get_meta("detectors")
    assert round_order_tick.get_meta("observables") == baseline_tick.get_meta("observables")

    def guppy_syndrome_host_ids(plan_id: str) -> list[str]:
        source = generate_guppy_source(
            patch,
            num_rounds=1,
            ancilla_budget=2,
            check_plan=plan_id,
        )
        return [
            str(metadata["host_id"])
            for _, metadata in _packed_trace_metadata_records(source)
            if metadata.get("source_kind") == "szz_host"
            and str(metadata.get("host_id", "")).startswith("szz:syndrome_extraction:")
        ]

    baseline_hosts = guppy_syndrome_host_ids(baseline_plan)
    round_order_hosts = guppy_syndrome_host_ids(round_order_plan)
    assert round_order_hosts != baseline_hosts
    assert round_order_hosts[:4] == [
        "szz:syndrome_extraction:r1:X0:d0:SZZ",
        "szz:syndrome_extraction:r1:Z3:d5:SZZDG",
        "szz:syndrome_extraction:r4:X0:d1:SZZDG",
        "szz:syndrome_extraction:r4:Z3:d2:SZZ",
    ]


def test_direct_surface_renderers_reject_plan_basis_mismatch() -> None:
    from pecos.qec.surface import SurfacePatch
    from pecos.qec.surface.circuit_builder import generate_tick_circuit_from_patch
    from pecos.qec.surface.decode import build_memory_circuit

    patch = SurfacePatch.create(distance=3)

    with pytest.raises(ValueError, match="conflicts with check_plan"):
        generate_tick_circuit_from_patch(
            patch,
            num_rounds=1,
            interaction_basis="cx",
            check_plan="szz_current_v1",
        )

    with pytest.raises(ValueError, match="conflicts with check_plan"):
        build_memory_circuit(
            patch=patch,
            rounds=1,
            interaction_basis="cx",
            check_plan="szz_current_v1",
        )


def test_native_sampler_records_resolved_check_plan() -> None:
    from pecos.qec.surface import NoiseModel, SurfacePatch, build_native_sampler

    patch = SurfacePatch.create(distance=3)
    sampler = build_native_sampler(
        patch,
        num_rounds=1,
        noise=NoiseModel(p2=0.001),
        check_plan="szz_current_v1",
        sampling_model="influence_dem",
    )

    assert sampler.interaction_basis == "szz"
    assert sampler.check_plan == "szz_current_v1"
    assert sampler.resolved_check_plan is not None
    assert sampler.resolved_check_plan["semantic_content"]["interaction_basis"] == "szz"
    assert len(sampler.resolved_check_plan_hash) == 64
