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
identical batch sequences from both call sites.

The two functions are intentionally pure (no circuit object created)
so neither consumer pulls in the other's dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.qec.surface.geometry import SurfacePatch


def normalize_ancilla_budget(total_ancilla: int, ancilla_budget: int | None) -> int:
    """Clamp an ancilla budget to the valid range for a patch.

    ``None`` collapses to the unconstrained ``total_ancilla``. A budget
    ``>= total_ancilla`` clamps to ``total_ancilla`` so callers
    requesting "no constraint" via either ``None`` or a large
    integer resolve to the same effective budget. ``< 1`` is rejected
    fail-loud.
    """
    if ancilla_budget is None:
        return total_ancilla

    if ancilla_budget < 1:
        msg = f"ancilla_budget must be >= 1, got {ancilla_budget}"
        raise ValueError(msg)

    return min(ancilla_budget, total_ancilla)


def batched_stabilizers(
    patch: SurfacePatch,
    ancilla_budget: int,
) -> list[list[tuple[str, int]]]:
    """Partition stabilizers into ancilla-reuse batches.

    Returns a list of batches, each a list of ``(stab_type, stab_idx)``
    pairs where ``stab_type`` is ``"X"`` or ``"Z"`` and ``stab_idx`` is
    the patch-internal stabilizer index. Batches are at most
    ``ancilla_budget`` stabilizers each; within each batch every
    stabilizer is measured concurrently using one ancilla qubit.

    The stabilizer order is **load-bearing** and shared between the
    abstract circuit and the Guppy emitter: ascending stabilizer index,
    X before Z on ties. Any change here will diverge the abstract DEM
    from the traced-Guppy DEM in the Selene parity tests; preserve it.
    """
    geom = patch.geometry
    stabilizers = [("X", stab.index) for stab in geom.x_stabilizers]
    stabilizers.extend(("Z", stab.index) for stab in geom.z_stabilizers)
    stabilizers.sort(key=lambda stab: (stab[1], 0 if stab[0] == "X" else 1))

    return [stabilizers[start : start + ancilla_budget] for start in range(0, len(stabilizers), ancilla_budget)]
