"""Shared helpers for ancilla-budget reasoning across surface paths.

Both the abstract surface-circuit builder
(``pecos.qec.surface.circuit_builder``) and the Guppy emitter
(``pecos.guppy.surface``) need to agree, byte-for-byte, on how
stabilizers are partitioned into ancilla-reuse batches. Otherwise the
abstract reference TickCircuit and the traced Guppy program produce
different measurement orders, the detector record offsets the caller
passes reference the wrong measurements, and the DEM is silently
wrong.

Keeping the partitioning logic in this single helper -- imported by
both consumers -- is the only source of truth. A unit test pins
concrete expected batch sequences for small ``(distance, budget)``
combinations (see
``tests/qec/surface/test_ancilla_batching.py``) so a regression in
the partitioning policy itself fails fast, independent of any DEM-
level oracle.

The two functions are intentionally pure (no circuit object created)
so neither consumer pulls in the other's dependencies.
"""

from __future__ import annotations

from math import ceil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pecos.qec.surface.geometry import SurfacePatch


DEFAULT_ANCILLA_SCHEDULE = "default"
BALANCED_DATA_ANCILLA_SCHEDULE = "balanced-data-v1"
SUPPORTED_ANCILLA_SCHEDULES = frozenset(
    {DEFAULT_ANCILLA_SCHEDULE, BALANCED_DATA_ANCILLA_SCHEDULE},
)


def normalize_ancilla_schedule(ancilla_schedule: str | None = None) -> str:
    """Return the canonical named ancilla-reuse schedule.

    ``None`` is the historical default batching policy.  Non-default policies
    must be explicit in source-level check-plan metadata so DEM caches and
    traced programs cannot accidentally collide with the legacy schedule.
    """
    if ancilla_schedule is None:
        return DEFAULT_ANCILLA_SCHEDULE
    normalized = str(ancilla_schedule).lower().replace("_", "-")
    if normalized not in SUPPORTED_ANCILLA_SCHEDULES:
        msg = f"ancilla_schedule must be one of {sorted(SUPPORTED_ANCILLA_SCHEDULES)}, got {ancilla_schedule!r}"
        raise ValueError(msg)
    return normalized


def normalize_ancilla_budget(total_ancilla: int, ancilla_budget: int | None) -> int:
    """Clamp an ancilla budget to the valid range for a patch.

    ``None`` collapses to the unconstrained ``total_ancilla``. A budget
    ``>= total_ancilla`` clamps to ``total_ancilla`` so callers
    requesting "no constraint" via either ``None`` or a large integer
    resolve to the same effective budget. ``< 1`` is rejected fail-loud.

    Non-``int`` (including ``bool``, ``float``) is rejected fail-loud
    so the public ``ancilla_budget`` kwarg has a strict integer
    contract -- avoiding silently-wrong cache keys or qubit counts.
    """
    if ancilla_budget is None:
        return total_ancilla

    # Reject bool first (bool is a subclass of int in Python).
    if isinstance(ancilla_budget, bool) or not isinstance(ancilla_budget, int):
        msg = f"ancilla_budget must be int or None, got {type(ancilla_budget).__name__}"
        raise TypeError(msg)

    if ancilla_budget < 1:
        msg = f"ancilla_budget must be >= 1, got {ancilla_budget}"
        raise ValueError(msg)

    return min(ancilla_budget, total_ancilla)


def batched_stabilizers(
    patch: SurfacePatch,
    ancilla_budget: int,
    *,
    ancilla_schedule: str | None = None,
) -> list[list[tuple[str, int]]]:
    """Partition stabilizers into ancilla-reuse batches.

    Returns a list of batches, each a list of ``(stab_type, stab_idx)``
    pairs where ``stab_type`` is ``"X"`` or ``"Z"`` and ``stab_idx`` is
    the patch-internal stabilizer index. Batches are at most
    ``ancilla_budget`` stabilizers each; within each batch every
    stabilizer is measured concurrently using one ancilla qubit.

    The default stabilizer order is **load-bearing** production semantics
    shared by the abstract circuit and the Guppy emitter: ascending stabilizer
    index, X before Z on ties. Note the traced-vs-traced Selene parity tests
    cannot catch a regression here -- both sides import this one helper, so a
    policy change moves them together. The concrete batch-order and
    source-level CX-emission pins
    (``tests/qec/surface/test_ancilla_batching.py``) are what actually guard
    this order; preserve it.

    ``ancilla_schedule="balanced-data-v1"`` is an explicit non-default policy
    for constrained-ancilla programs. It greedily spreads stabilizer supports
    across each batch so data qubits see check interactions more uniformly
    through the batched sequence. This does not change result tags or detector
    semantics; callers must still record the chosen schedule in check-plan
    metadata so cached DEMs do not collide with the default schedule.

    ``ancilla_budget`` is validated through
    :func:`normalize_ancilla_budget` (rejects ``None``, ``bool``,
    ``float``, ``str``, ``< 1``; clamps ``>= total_ancilla``) so direct
    callers of this helper get the same fail-loud guarantees as the
    public ``ancilla_budget`` API surface, not an opaque ``range()`` or
    silent-empty failure.
    """
    schedule = normalize_ancilla_schedule(ancilla_schedule)
    geom = patch.geometry
    total_ancilla = len(geom.x_stabilizers) + len(geom.z_stabilizers)
    effective_budget = normalize_ancilla_budget(total_ancilla, ancilla_budget)

    stabilizers = _canonical_stabilizer_order(patch)
    if schedule == BALANCED_DATA_ANCILLA_SCHEDULE:
        return _balanced_data_batches(patch, stabilizers, effective_budget)

    return [stabilizers[start : start + effective_budget] for start in range(0, len(stabilizers), effective_budget)]


def _canonical_stabilizer_order(patch: SurfacePatch) -> list[tuple[str, int]]:
    geom = patch.geometry
    stabilizers = [("X", stab.index) for stab in geom.x_stabilizers]
    stabilizers.extend(("Z", stab.index) for stab in geom.z_stabilizers)
    stabilizers.sort(key=_canonical_stabilizer_sort_key)
    return stabilizers


def _canonical_stabilizer_sort_key(stabilizer: tuple[str, int]) -> tuple[int, int]:
    stab_type, stab_idx = stabilizer
    return (stab_idx, 0 if stab_type == "X" else 1)


def _balanced_data_batches(
    patch: SurfacePatch,
    stabilizers: list[tuple[str, int]],
    effective_budget: int,
) -> list[list[tuple[str, int]]]:
    """Greedily spread data-qubit supports inside each constrained batch."""
    if effective_budget >= len(stabilizers):
        return [stabilizers]

    by_stabilizer = _stabilizer_support_lookup(patch)
    canonical_order = {stabilizer: index for index, stabilizer in enumerate(stabilizers)}
    remaining = set(stabilizers)
    batches: list[list[tuple[str, int]]] = []
    for batch_index, target_size in enumerate(_balanced_batch_sizes(len(stabilizers), effective_budget)):
        batch: list[tuple[str, int]] = []
        touched_data: set[int] = set()
        x_count = 0
        z_count = 0

        while remaining and len(batch) < target_size:
            score_state = (frozenset(touched_data), x_count, z_count, batch_index)

            def score(
                stabilizer: tuple[str, int],
                *,
                state: tuple[frozenset[int], int, int, int] = score_state,
            ) -> tuple[int, int, float, float, int]:
                bound_touched_data, bound_x_count, bound_z_count, bound_batch_index = state
                support, row, col = by_stabilizer[stabilizer]
                overlap = sum(data_qubit in bound_touched_data for data_qubit in support)
                next_x = bound_x_count + (1 if stabilizer[0] == "X" else 0)
                next_z = bound_z_count + (1 if stabilizer[0] == "Z" else 0)
                type_imbalance = abs(next_x - next_z)
                # Alternate the spatial sweep direction between batches so the
                # deterministic tie-break does not repeatedly privilege the
                # same edge of the patch.
                row_key = row if bound_batch_index % 2 == 0 else -row
                col_key = col if bound_batch_index % 2 == 0 else -col
                return (overlap, type_imbalance, row_key, col_key, canonical_order[stabilizer])

            selected = min(remaining, key=score)
            remaining.remove(selected)
            batch.append(selected)
            support, _row, _col = by_stabilizer[selected]
            touched_data.update(support)
            if selected[0] == "X":
                x_count += 1
            else:
                z_count += 1

        batches.append(batch)

    return batches


def _balanced_batch_sizes(total: int, effective_budget: int) -> list[int]:
    """Return near-equal batch sizes, each at most ``effective_budget``."""
    num_batches = ceil(total / effective_budget)
    base_size, remainder = divmod(total, num_batches)
    return [base_size + (1 if index < remainder else 0) for index in range(num_batches)]


def _stabilizer_support_lookup(
    patch: SurfacePatch,
) -> dict[tuple[str, int], tuple[tuple[int, ...], float, float]]:
    geom = patch.geometry
    lookup: dict[tuple[str, int], tuple[tuple[int, ...], float, float]] = {}

    def add(stabilizers: Iterable[object], stab_type: str) -> None:
        for stabilizer in stabilizers:
            support = tuple(int(q) for q in stabilizer.data_qubits)
            positions = [geom.id_to_pos[q] for q in support]
            row = sum(pos[0] for pos in positions) / len(positions)
            col = sum(pos[1] for pos in positions) / len(positions)
            lookup[(stab_type, int(stabilizer.index))] = (support, row, col)

    add(geom.x_stabilizers, "X")
    add(geom.z_stabilizers, "Z")
    return lookup
