# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Surface check-plan metadata tests."""

from __future__ import annotations

import hashlib

import pytest


def test_check_plan_default_resolves_to_cx_metadata() -> None:
    from pecos.qec.surface._check_plan import canonical_check_plan_json, resolve_surface_check_plan

    plan = resolve_surface_check_plan()

    assert plan.plan_id == "cx_standard_v1"
    assert plan.interaction_basis == "cx"
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


def test_check_plan_and_interaction_basis_mismatch_fails_loudly() -> None:
    from pecos.qec.surface._check_plan import resolve_surface_check_plan

    with pytest.raises(ValueError, match="conflicts with check_plan"):
        resolve_surface_check_plan(
            interaction_basis="cx",
            check_plan="szz_current_v1",
        )


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
