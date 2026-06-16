# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Resolved surface-code check-plan metadata."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

CHECK_PLAN_METADATA_FORMAT = "pecos.surface.check_plan"
CHECK_PLAN_METADATA_VERSION = 1
CHECK_PLAN_HASH_ALGORITHM = "sha256"
CHECK_PLAN_HASH_SERIALIZATION = "canonical-json-v1"
CURRENT_SURFACE_CHECK_PLAN_RENDERER = "pecos.surface.current_renderer_v1"

_DEFAULT_CHECK_PLAN_BY_BASIS = {
    "cx": "cx_standard_v1",
    "szz": "szz_current_v1",
}


def _normalize_interaction_basis_name(interaction_basis: str) -> str:
    normalized = interaction_basis.lower()
    if normalized not in _DEFAULT_CHECK_PLAN_BY_BASIS:
        msg = f"interaction_basis must be 'cx' or 'szz', got {interaction_basis!r}"
        raise ValueError(msg)
    return normalized


def _normalize_check_plan_id(check_plan: str) -> str:
    normalized = check_plan.lower()
    if normalized not in _PLAN_SEMANTICS:
        msg = f"unknown check_plan {check_plan!r}; expected one of {sorted(_PLAN_SEMANTICS)}"
        raise ValueError(msg)
    return normalized


_PLAN_SEMANTICS: dict[str, dict[str, Any]] = {
    "cx_standard_v1": {
        "plan_id": "cx_standard_v1",
        "interaction_basis": "cx",
        "synthesis_identity": {
            "family": "cx",
            "szz_phase_pattern": "none",
            "interaction_order": "pecos-default",
            "ancilla_schedule": "default",
        },
        "schedule": {
            "round_policy": "constant",
            "site_policy": "global",
            "edge_order": "current_surface_cnot_schedule_v1",
        },
        "x_check": {
            "template": "current_cx_x_check_v1",
            "measurement_sign_policy": "none",
        },
        "z_check": {
            "template": "current_cx_z_check_v1",
            "measurement_sign_policy": "none",
        },
        "prefix_policy": "none",
    },
    "szz_current_v1": {
        "plan_id": "szz_current_v1",
        "interaction_basis": "szz",
        "synthesis_identity": {
            "family": "szz",
            "szz_phase_pattern": "standard",
            "interaction_order": "pecos-default",
            "ancilla_schedule": "default",
        },
        "schedule": {
            "round_policy": "constant",
            "site_policy": "global",
            "edge_order": "current_surface_cnot_schedule_v1",
        },
        "x_check": {
            "template": "current_szz_x_check_v1",
            "sign_policy": "default_szz_sign_vector_v1",
            "residual_policy": "per_touch_compensated",
            "measurement_sign_policy": "explicit_template_metadata",
        },
        "z_check": {
            "template": "current_szz_z_check_v1",
            "sign_policy": "default_szz_sign_vector_v1",
            "residual_policy": "per_touch_compensated",
            "measurement_sign_policy": "explicit_template_metadata",
        },
        "prefix_policy": "forward_flow_virtual_z_v1",
    },
}


def canonical_check_plan_json(value: dict[str, Any]) -> str:
    """Serialize check-plan metadata with stable cross-platform bytes."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def surface_check_plan_ids() -> tuple[str, ...]:
    """Return known surface check-plan IDs in deterministic order."""
    return tuple(sorted(_PLAN_SEMANTICS))


def default_surface_check_plan_id(interaction_basis: str | None = None) -> str:
    """Return the default check-plan ID for a two-qubit interaction basis."""
    if interaction_basis is None:
        return _DEFAULT_CHECK_PLAN_BY_BASIS["cx"]
    return _DEFAULT_CHECK_PLAN_BY_BASIS[_normalize_interaction_basis_name(interaction_basis)]


def _semantic_hash(semantic_content: dict[str, Any]) -> str:
    encoded = canonical_check_plan_json(semantic_content).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ResolvedSurfaceCheckPlan:
    """Internal resolved check-plan metadata for current surface-memory presets."""

    plan_id: str
    interaction_basis: str
    synthesis_identity: dict[str, Any]
    semantic_content: dict[str, Any]
    resolved_metadata: dict[str, Any]
    resolved_hash: str


def resolve_surface_check_plan(
    *,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
) -> ResolvedSurfaceCheckPlan:
    """Resolve the public check-plan selector to deterministic metadata.

    ``check_plan`` is the source of truth. ``interaction_basis`` remains a
    backward-compatible default-plan selector and must agree when both are
    provided.
    """
    normalized_basis = None if interaction_basis is None else _normalize_interaction_basis_name(interaction_basis)
    if check_plan is None:
        plan_id = _DEFAULT_CHECK_PLAN_BY_BASIS[normalized_basis or "cx"]
    else:
        plan_id = _normalize_check_plan_id(check_plan)

    semantic_content = json.loads(canonical_check_plan_json(_PLAN_SEMANTICS[plan_id]))
    plan_basis = str(semantic_content["interaction_basis"])
    if normalized_basis is not None and normalized_basis != plan_basis:
        msg = (
            f"interaction_basis={normalized_basis!r} conflicts with "
            f"check_plan={plan_id!r}, which uses interaction_basis={plan_basis!r}"
        )
        raise ValueError(msg)

    resolved_hash = _semantic_hash(semantic_content)
    resolved_metadata = {
        "format": CHECK_PLAN_METADATA_FORMAT,
        "metadata_version": CHECK_PLAN_METADATA_VERSION,
        "hash_algorithm": CHECK_PLAN_HASH_ALGORITHM,
        "hash_serialization": CHECK_PLAN_HASH_SERIALIZATION,
        "semantic_content": semantic_content,
    }
    return ResolvedSurfaceCheckPlan(
        plan_id=plan_id,
        interaction_basis=plan_basis,
        synthesis_identity=dict(semantic_content["synthesis_identity"]),
        semantic_content=semantic_content,
        resolved_metadata=resolved_metadata,
        resolved_hash=resolved_hash,
    )


def require_current_surface_check_plan_renderer(
    resolved_plan: ResolvedSurfaceCheckPlan,
    *,
    context: str,
) -> None:
    """Fail if the current source renderers cannot realize ``resolved_plan``.

    Check plans are intended to be source-level circuit contracts, not metadata
    labels pasted onto whatever circuit the old ``interaction_basis`` path
    happened to emit. Keep this guard close to circuit generation until each new
    plan has an explicit renderer.
    """
    semantic = resolved_plan.semantic_content
    synthesis = resolved_plan.synthesis_identity
    schedule = semantic.get("schedule", {})

    expected_synthesis = {
        "cx": {
            "family": "cx",
            "szz_phase_pattern": "none",
            "interaction_order": "pecos-default",
            "ancilla_schedule": "default",
        },
        "szz": {
            "family": "szz",
            "szz_phase_pattern": "standard",
            "interaction_order": "pecos-default",
            "ancilla_schedule": "default",
        },
    }[resolved_plan.interaction_basis]
    expected_schedule = {
        "round_policy": "constant",
        "site_policy": "global",
        "edge_order": "current_surface_cnot_schedule_v1",
    }
    if synthesis != expected_synthesis or schedule != expected_schedule:
        msg = (
            f"{context} cannot realize check_plan={resolved_plan.plan_id!r} "
            f"with {CURRENT_SURFACE_CHECK_PLAN_RENDERER}; synthesis_identity={synthesis!r}, "
            f"schedule={schedule!r}"
        )
        raise NotImplementedError(msg)

    if resolved_plan.interaction_basis == "cx":
        expected_checks = {
            "prefix_policy": "none",
            "x_check": {
                "template": "current_cx_x_check_v1",
                "measurement_sign_policy": "none",
            },
            "z_check": {
                "template": "current_cx_z_check_v1",
                "measurement_sign_policy": "none",
            },
        }
    else:
        expected_checks = {
            "prefix_policy": "forward_flow_virtual_z_v1",
            "x_check": {
                "template": "current_szz_x_check_v1",
                "sign_policy": "default_szz_sign_vector_v1",
                "residual_policy": "per_touch_compensated",
                "measurement_sign_policy": "explicit_template_metadata",
            },
            "z_check": {
                "template": "current_szz_z_check_v1",
                "sign_policy": "default_szz_sign_vector_v1",
                "residual_policy": "per_touch_compensated",
                "measurement_sign_policy": "explicit_template_metadata",
            },
        }

    actual_checks = {
        "prefix_policy": semantic.get("prefix_policy"),
        "x_check": semantic.get("x_check"),
        "z_check": semantic.get("z_check"),
    }
    if actual_checks != expected_checks:
        msg = (
            f"{context} cannot realize check_plan={resolved_plan.plan_id!r} "
            f"with {CURRENT_SURFACE_CHECK_PLAN_RENDERER}; check templates={actual_checks!r}"
        )
        raise NotImplementedError(msg)
