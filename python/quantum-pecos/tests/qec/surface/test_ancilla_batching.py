# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Tests for the shared ancilla-batching helper.

This is the single source of truth for stabilizer-batch ordering used
by both the abstract surface-circuit builder
(``pecos.qec.surface.circuit_builder``) and the Guppy emitter
(``pecos.guppy.surface``). The byte-identical traced-vs-traced surface
DEM oracle in ``tests/qec/test_from_guppy_dem.py`` exercises this
helper indirectly, but a regression in the partitioning *policy*
itself (e.g. someone changes the sort key) could pass that oracle
spuriously because both sides share the same shared helper. Concrete
expected-output pins below catch that case directly.
"""

from __future__ import annotations

import pytest
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface._ancilla_batching import (
    batched_stabilizers,
    normalize_ancilla_budget,
)

# --- normalize_ancilla_budget -----------------------------------------------


@pytest.mark.parametrize(
    ("total", "budget", "expected"),
    [
        (8, None, 8),  # None means "no constraint"
        (8, 8, 8),  # exact match
        (8, 9, 8),  # >= total collapses to total
        (8, 999, 8),  # large budget collapses to total
        (8, 1, 1),  # minimum valid
        (8, 4, 4),  # interior
    ],
)
def test_normalize_ancilla_budget_clamps(total: int, budget: int | None, expected: int) -> None:
    assert normalize_ancilla_budget(total, budget) == expected


def test_normalize_ancilla_budget_rejects_zero_and_negative() -> None:
    with pytest.raises(ValueError, match=r"must be >= 1"):
        normalize_ancilla_budget(8, 0)
    with pytest.raises(ValueError, match=r"must be >= 1"):
        normalize_ancilla_budget(8, -1)


def test_normalize_ancilla_budget_rejects_non_int() -> None:
    """Public ``ancilla_budget`` kwarg has a strict ``int | None`` contract.

    bool is a Python subclass of int but a separate semantic type; rejecting
    it explicitly avoids ``True``-as-``1`` silently working, which would mask
    caller-side bugs."""
    with pytest.raises(TypeError, match=r"must be int or None, got bool"):
        normalize_ancilla_budget(8, True)
    with pytest.raises(TypeError, match=r"must be int or None, got float"):
        normalize_ancilla_budget(8, 1.5)
    with pytest.raises(TypeError, match=r"must be int or None, got str"):
        normalize_ancilla_budget(8, "1")


# --- batched_stabilizers (concrete sequences) -------------------------------


def test_batched_stabilizers_d3_budget1_one_stabilizer_per_batch() -> None:
    """Budget=1 produces one stabilizer per batch, alternating X/Z by
    ascending index per the shared sort key. Pinning this concrete order
    catches "shared batching policy regressed" independent of any DEM-
    level oracle."""
    patch = SurfacePatch.create(distance=3)
    batches = batched_stabilizers(patch, 1)
    assert batches == [
        [("X", 0)],
        [("Z", 0)],
        [("X", 1)],
        [("Z", 1)],
        [("X", 2)],
        [("Z", 2)],
        [("X", 3)],
        [("Z", 3)],
    ]


def test_batched_stabilizers_d3_budget2_pairs_xz_by_index() -> None:
    """Budget=2 pairs (X_k, Z_k) per batch for ascending k."""
    patch = SurfacePatch.create(distance=3)
    batches = batched_stabilizers(patch, 2)
    assert batches == [
        [("X", 0), ("Z", 0)],
        [("X", 1), ("Z", 1)],
        [("X", 2), ("Z", 2)],
        [("X", 3), ("Z", 3)],
    ]


def test_batched_stabilizers_full_budget_one_batch() -> None:
    """Budget == total_ancilla collapses to a single batch containing
    every stabilizer in the canonical sort order."""
    patch = SurfacePatch.create(distance=3)
    total = len(patch.geometry.x_stabilizers) + len(patch.geometry.z_stabilizers)
    batches = batched_stabilizers(patch, total)
    assert len(batches) == 1
    assert batches[0] == [
        ("X", 0),
        ("Z", 0),
        ("X", 1),
        ("Z", 1),
        ("X", 2),
        ("Z", 2),
        ("X", 3),
        ("Z", 3),
    ]


def test_batched_stabilizers_distance_5_budget_3_covers_all_stabilizers() -> None:
    """For a slightly bigger patch, every stabilizer appears exactly once
    across the returned batches, with batch sizes ``<= budget``."""
    patch = SurfacePatch.create(distance=5)
    total = len(patch.geometry.x_stabilizers) + len(patch.geometry.z_stabilizers)
    batches = batched_stabilizers(patch, 3)

    assert all(len(batch) <= 3 for batch in batches)

    flat = [pair for batch in batches for pair in batch]
    assert len(flat) == total
    assert len(set(flat)) == total  # no duplicates


# --- batched_stabilizers input validation ---------------------------------


def test_batched_stabilizers_rejects_invalid_budget_directly() -> None:
    """``batched_stabilizers`` validates its own ``ancilla_budget`` (routes
    through ``normalize_ancilla_budget``) rather than producing an opaque
    ``range()`` error or a silent-empty failure on ``0`` / non-int input.
    Closes the self-review's A2 finding."""
    patch = SurfacePatch.create(distance=3)
    with pytest.raises(ValueError, match=r"must be >= 1"):
        batched_stabilizers(patch, 0)
    with pytest.raises(ValueError, match=r"must be >= 1"):
        batched_stabilizers(patch, -2)
    with pytest.raises(TypeError, match=r"must be int or None"):
        batched_stabilizers(patch, True)
    with pytest.raises(TypeError, match=r"must be int or None"):
        batched_stabilizers(patch, 1.5)


def test_batched_stabilizers_clamps_oversized_budget() -> None:
    """A budget larger than ``total_ancilla`` clamps to one big batch,
    matching ``normalize_ancilla_budget`` behavior. Direct callers get the
    same clamping the public API surface gets."""
    patch = SurfacePatch.create(distance=3)
    total = len(patch.geometry.x_stabilizers) + len(patch.geometry.z_stabilizers)
    huge = batched_stabilizers(patch, 10**6)
    assert len(huge) == 1
    assert len(huge[0]) == total


# --- D1: pin emitted CX sequences for the constrained Guppy codegen --------
# The byte-identical traced-vs-traced DEM oracle and the lowered-qubit-stream
# invariant catch many constrained-codegen errors, but not a wrong-CX-order /
# wrong-CX-control / dropped-CX bug inside the emitter (the lowered Selene
# trace uses RZZ + surrounding rotations, not raw CX, so the trace doesn't
# expose the emitted CX shape directly). These tests pin the literal CX
# emission at the **source** level so a regression in
# ``generate_guppy_source``'s per-batch CX restriction fails fast,
# independent of any DEM-level oracle.


def _emitted_cx_lines(distance: int, ancilla_budget: int | None) -> list[str]:
    """Return the ``cx(...)`` lines emitted in the syndrome_extraction
    function for a given (distance, budget)."""
    import re

    from pecos.guppy.surface import generate_surface_code_module

    src = generate_surface_code_module(distance, ancilla_budget=ancilla_budget)
    in_se = False
    cx_lines: list[str] = []
    for line in src.split("\n"):
        if line.startswith("def syndrome_extraction"):
            in_se = True
            continue
        # Stop at the next top-level def or @ decorator (next function).
        if in_se and line and not line.startswith(" ") and not line.startswith("#"):
            break
        if in_se:
            m = re.match(r"^\s*(cx\([^)]+\))", line)
            if m:
                cx_lines.append(m.group(1))
    return cx_lines


def test_constrained_d3_budget1_emits_expected_cx_sequence() -> None:
    """Catches wrong-CX-order / wrong-control / dropped-CX bugs in the
    constrained emitter that the DEM-level and trace-level oracles miss."""
    assert _emitted_cx_lines(3, 1) == [
        "cx(_a_b0_p0, surf.data[1])",
        "cx(_a_b0_p0, surf.data[0])",
        "cx(surf.data[3], _a_b1_p0)",
        "cx(surf.data[6], _a_b1_p0)",
        "cx(_a_b2_p0, surf.data[2])",
        "cx(_a_b2_p0, surf.data[1])",
        "cx(_a_b2_p0, surf.data[5])",
        "cx(_a_b2_p0, surf.data[4])",
        "cx(surf.data[1], _a_b3_p0)",
        "cx(surf.data[4], _a_b3_p0)",
        "cx(surf.data[0], _a_b3_p0)",
        "cx(surf.data[3], _a_b3_p0)",
        "cx(_a_b4_p0, surf.data[4])",
        "cx(_a_b4_p0, surf.data[3])",
        "cx(_a_b4_p0, surf.data[7])",
        "cx(_a_b4_p0, surf.data[6])",
        "cx(surf.data[5], _a_b5_p0)",
        "cx(surf.data[8], _a_b5_p0)",
        "cx(surf.data[4], _a_b5_p0)",
        "cx(surf.data[7], _a_b5_p0)",
        "cx(_a_b6_p0, surf.data[8])",
        "cx(_a_b6_p0, surf.data[7])",
        "cx(surf.data[2], _a_b7_p0)",
        "cx(surf.data[5], _a_b7_p0)",
    ]


def test_constrained_d3_budget2_emits_expected_cx_sequence() -> None:
    """Pins the budget=2 batched CX schedule (pairs X_k with Z_k each batch,
    CXs filtered to that batch's stabilizers across 4 schedule rounds)."""
    assert _emitted_cx_lines(3, 2) == [
        "cx(surf.data[3], _a_b0_p1)",
        "cx(surf.data[6], _a_b0_p1)",
        "cx(_a_b0_p0, surf.data[1])",
        "cx(_a_b0_p0, surf.data[0])",
        "cx(_a_b1_p0, surf.data[2])",
        "cx(surf.data[1], _a_b1_p1)",
        "cx(_a_b1_p0, surf.data[1])",
        "cx(surf.data[4], _a_b1_p1)",
        "cx(_a_b1_p0, surf.data[5])",
        "cx(surf.data[0], _a_b1_p1)",
        "cx(_a_b1_p0, surf.data[4])",
        "cx(surf.data[3], _a_b1_p1)",
        "cx(_a_b2_p0, surf.data[4])",
        "cx(surf.data[5], _a_b2_p1)",
        "cx(_a_b2_p0, surf.data[3])",
        "cx(surf.data[8], _a_b2_p1)",
        "cx(_a_b2_p0, surf.data[7])",
        "cx(surf.data[4], _a_b2_p1)",
        "cx(_a_b2_p0, surf.data[6])",
        "cx(surf.data[7], _a_b2_p1)",
        "cx(_a_b3_p0, surf.data[8])",
        "cx(_a_b3_p0, surf.data[7])",
        "cx(surf.data[2], _a_b3_p1)",
        "cx(surf.data[5], _a_b3_p1)",
    ]
