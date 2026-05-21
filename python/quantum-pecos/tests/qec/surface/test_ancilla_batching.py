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
