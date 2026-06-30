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

from itertools import permutations

import pytest
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface._ancilla_batching import (
    BALANCED_DATA_ANCILLA_SCHEDULE,
    batched_stabilizers,
    normalize_ancilla_budget,
    normalize_ancilla_schedule,
)
from pecos.qec.surface.schedule import compute_cnot_schedule

StabilizerKey = tuple[str, int]
TouchGapMetrics = tuple[int, int]


def _touch_gap_metrics_for_batches(
    patch: SurfacePatch,
    batches: list[list[StabilizerKey]],
    *,
    round_order: tuple[int, int, int, int] = (0, 1, 2, 3),
) -> TouchGapMetrics:
    cnot_rounds = compute_cnot_schedule(patch)
    ordered_rounds = [cnot_rounds[index] for index in round_order]
    events_by_data: dict[int, list[int]] = {}
    time = 0
    for batch in batches:
        batch_keys = set(batch)
        for round_gates in ordered_rounds:
            for stab_type, stab_idx, data_qubit in round_gates:
                if (stab_type, stab_idx) in batch_keys:
                    events_by_data.setdefault(data_qubit, []).append(time)
            time += 1
    period = time
    worst_gap = 0
    squared_gap_sum = 0
    for touch_times in events_by_data.values():
        touch_times.sort()
        gaps = [
            touch_times[index + 1] - touch_times[index]
            for index in range(len(touch_times) - 1)
        ]
        gaps.append(period + touch_times[0] - touch_times[-1])
        worst_gap = max(worst_gap, *gaps)
        squared_gap_sum += sum(gap * gap for gap in gaps)
    return worst_gap, squared_gap_sum

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


# --- normalize_ancilla_schedule ---------------------------------------------


def test_normalize_ancilla_schedule_accepts_named_policies() -> None:
    assert normalize_ancilla_schedule(None) == "default"
    assert normalize_ancilla_schedule("default") == "default"
    assert normalize_ancilla_schedule("balanced_data_v1") == BALANCED_DATA_ANCILLA_SCHEDULE
    assert normalize_ancilla_schedule("balanced-data-v1") == BALANCED_DATA_ANCILLA_SCHEDULE


def test_normalize_ancilla_schedule_rejects_unknown_policy() -> None:
    with pytest.raises(ValueError, match=r"ancilla_schedule must be one of"):
        normalize_ancilla_schedule("row-scan")


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


def test_balanced_data_schedule_is_explicit_and_deterministic() -> None:
    """The balanced schedule is a named non-default policy, not a change to
    the legacy batching semantics."""
    patch = SurfacePatch.create(distance=3)

    assert batched_stabilizers(patch, 2) == [
        [("X", 0), ("Z", 0)],
        [("X", 1), ("Z", 1)],
        [("X", 2), ("Z", 2)],
        [("X", 3), ("Z", 3)],
    ]
    assert batched_stabilizers(
        patch,
        2,
        ancilla_schedule=BALANCED_DATA_ANCILLA_SCHEDULE,
    ) == [
        [("X", 0), ("Z", 3)],
        [("X", 3), ("Z", 0)],
        [("Z", 1), ("Z", 2)],
        [("X", 2), ("X", 1)],
    ]


def test_balanced_data_schedule_spreads_d9_a17_batches() -> None:
    """The d=9/a17 target gets equal-size batches and no data qubit whose
    four adjacent checks all live in one batch."""
    patch = SurfacePatch.create(distance=9)
    batches = batched_stabilizers(
        patch,
        17,
        ancilla_schedule=BALANCED_DATA_ANCILLA_SCHEDULE,
    )

    assert [len(batch) for batch in batches] == [16, 16, 16, 16, 16]

    batch_of = {stabilizer: batch_idx for batch_idx, batch in enumerate(batches) for stabilizer in batch}
    touches_by_data: dict[int, set[int]] = {}
    for stab in patch.geometry.x_stabilizers:
        for data_qubit in stab.data_qubits:
            touches_by_data.setdefault(data_qubit, set()).add(batch_of[("X", stab.index)])
    for stab in patch.geometry.z_stabilizers:
        for data_qubit in stab.data_qubits:
            touches_by_data.setdefault(data_qubit, set()).add(batch_of[("Z", stab.index)])

    assert min(len(batch_indices) for batch_indices in touches_by_data.values()) > 1


def test_balanced_data_d9_a17_batch_order_is_touch_gap_optimal() -> None:
    """Batch reordering alone does not improve the d=9/a17 touch-gap proxy.

    This pins the result of the first idle-hotspot scheduler probe: the next
    candidate should target lower-level touch/layer placement, not just
    permuting the already-balanced ancilla batches.
    """
    patch = SurfacePatch.create(distance=9)
    balanced_batches = batched_stabilizers(
        patch,
        17,
        ancilla_schedule=BALANCED_DATA_ANCILLA_SCHEDULE,
    )
    assert [len(batch) for batch in balanced_batches] == [16, 16, 16, 16, 16]

    baseline_metrics = _touch_gap_metrics_for_batches(patch, balanced_batches)
    best_permutation_metrics = min(
        _touch_gap_metrics_for_batches(
            patch,
            [balanced_batches[index] for index in permutation],
        )
        for permutation in permutations(range(len(balanced_batches)))
    )

    assert baseline_metrics == best_permutation_metrics


def test_balanced_data_d9_a17_round_order_touch_gap_tradeoff() -> None:
    """Round-order changes expose a real but correctness-gated tradeoff.

    The best max-gap round permutation improves the repeated-round data-touch
    gap proxy, but increases the squared-gap proxy. Promoting such a candidate
    needs a named check-plan schedule and hook/residual correctness checks, not
    an ad hoc source-order edit.
    """
    patch = SurfacePatch.create(distance=9)
    balanced_batches = batched_stabilizers(
        patch,
        17,
        ancilla_schedule=BALANCED_DATA_ANCILLA_SCHEDULE,
    )
    baseline_metrics = _touch_gap_metrics_for_batches(patch, balanced_batches)

    round_candidates = [
        (_touch_gap_metrics_for_batches(patch, balanced_batches, round_order=round_order), round_order)
        for round_order in permutations(range(4))
    ]
    best_max_gap = min(round_candidates)
    best_sumsq_with_baseline_max_gap = min(
        candidate
        for candidate in round_candidates
        if candidate[0][0] == baseline_metrics[0]
    )

    assert baseline_metrics == (14, 11292)
    assert best_max_gap == ((13, 11464), (3, 1, 0, 2))
    assert best_sumsq_with_baseline_max_gap == ((14, 11220), (1, 0, 3, 2))


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
