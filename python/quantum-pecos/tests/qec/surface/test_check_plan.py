# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Surface check-plan metadata tests."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest


def test_check_plan_default_resolves_to_cx_metadata() -> None:
    from pecos.qec.surface._check_plan import (
        canonical_check_plan_json,
        default_surface_check_plan_id,
        resolve_surface_check_plan,
        surface_check_plan_ids,
    )

    plan = resolve_surface_check_plan()

    assert surface_check_plan_ids() == ("cx_standard_v1", "szz_current_v1")
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
